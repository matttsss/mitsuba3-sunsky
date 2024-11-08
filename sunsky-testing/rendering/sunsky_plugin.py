import drjit as dr
import mitsuba as mi

from .sunsky_data import get_params, get_tgmm_table, NB_GAUSSIAN_PARAMS, sample_gaussian, tgmm_pdf

def inv_sin_theta(v: mi.Vector3f):
    return dr.rcp(dr.safe_sqrt(dr.maximum(v.x**2 + v.y**2, dr.epsilon(mi.Float)**2)))

class SunskyEmitter(mi.Emitter):

    def __init__(self, props):
        super().__init__(props)

        self.m_bsphere = mi.BoundingSphere3f(mi.Point3f(0), 1)
        self.m_surface_area = 4.0 * dr.pi

        # Sort variant specific variables
        if dr.hint(mi.is_spectral, mode="scalar"):
            nb_channels = 11
            dataset_name = "sunsky-testing/res/datasets/ssm_dataset_v2_spec"
            self.wavelengths = [320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720]
            self.wavelength_step = 40

        elif dr.hint(mi.is_rgb, mode="scalar"):
            nb_channels = 3
            dataset_name = "sunsky-testing/res/datasets/ssm_dataset_v2_rgb"

        # Get albedo as a Float
        albedo_t = props.get("albedo", 0.15)
        if isinstance(albedo_t, float):
            albedo = mi.Float([albedo_t] * nb_channels)


        elif isinstance(albedo_t, mi.Texture):
            si = dr.zeros(mi.SurfaceInteraction3f)

            if dr.hint(mi.is_spectral, mode="scalar"):
                albedo = [0.0] * nb_channels
                normalization = 0.0

                for i in range(nb_channels):
                    si.wavelengths = mi.Spectrum(self.wavelengths[i])
                    res = albedo_t.eval(si)[0]

                    normalization += res
                    albedo[i] = res

                albedo = mi.Float(dr.ravel(albedo)) / normalization

            elif dr.hint(mi.is_rgb, mode="scalar"):
                albedo = mi.Float(dr.ravel(albedo_t.eval(si)))


        else:
            raise RuntimeError("Invalid albedo type")

        turb = props.get("turbidity", 3)

        # Get sun direction / elevation
        self.to_world = props.get("to_world", mi.Transform4f(1))
        self.m_local_sun = dr.normalize(self.to_world.inverse() @ props.get("sun_direction"))

        cos_phi = mi.Frame3f.cos_phi(self.m_local_sun)
        sin_phi = mi.Frame3f.sin_phi(self.m_local_sun)

        self.sun_phi = dr.acos(cos_phi)
        self.sun_phi = dr.select(sin_phi >= 0, self.sun_phi, dr.two_pi - self.sun_phi)

        sun_eta = dr.pi / 2 - dr.acos(mi.Frame3f.cos_theta(self.m_local_sun))

        # Get luminance parameters
        _, database = mi.array_from_file_d(dataset_name + ".bin")
        _, database_rad = mi.array_from_file_d(dataset_name + "_rad.bin")
        self.m_params = get_params(database, turb, albedo, sun_eta)
        self.m_rad = get_params(database_rad, turb, albedo, sun_eta)

        # Get sampling parameters
        _, tgmm_tables = mi.array_from_file_f("sunsky-testing/res/datasets/tgmm_tables.bin")
        mis_weights, self.gaussians = get_tgmm_table(tgmm_tables, turb, sun_eta)

        self.gaussian_dist = mi.DiscreteDistribution(mis_weights)

        self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying

        dr.eval(self.m_params, self.m_rad, self.gaussians, self.gaussian_dist)

    def set_scene(self, scene):
        if scene.bbox().valid():
            self.m_bsphere = scene.bbox().bounding_sphere()
            self.m_bsphere.radius = dr.maximum(
                mi.math.RayEpsilon,
                self.m_bsphere.radius * (1 + mi.math.RayEpsilon))
        else:
            self.m_bsphere.radius = mi.BoundingSphere3f(0, mi.math.RayEpsilon)

        self.m_surface_area = 4.0 * dr.pi * self.m_bsphere.radius**2


    def render_channel(self, idx: mi.UInt32, cos_theta: mi.Float, cos_gamma: mi.Float, active: mi.Bool):
        coefs = dr.gather(mi.ArrayXf, self.m_params, idx, active, shape=(9, 1))

        gamma = dr.safe_acos(cos_gamma)
        cos_gamma_sqr = dr.square(cos_gamma)

        c1 = 1 + coefs[0] * dr.exp(coefs[1] / (cos_theta + 0.01))
        chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5)
        c2 = coefs[2] + coefs[3] * dr.exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr.safe_sqrt(cos_theta)

        return c1 * c2 * dr.gather(mi.Float, self.m_rad, idx, active)


    def eval(self, si, active=True):
        local_wi = self.to_world.inverse() @ si.wi
        cos_theta = mi.Frame3f.cos_theta(-local_wi)
        cos_gamma = dr.dot(self.m_local_sun, -local_wi)

        active &= cos_theta >= 0

        res = dr.zeros(mi.Spectrum)
        if dr.hint(mi.is_rgb, mode="scalar"):
            res[0] = self.render_channel(0, cos_theta, cos_gamma, active)
            res[1] = self.render_channel(1, cos_theta, cos_gamma, active)
            res[2] = self.render_channel(2, cos_theta, cos_gamma, active)

            res /= 106.856980

        else:
            normalized_wavelengths = (si.wavelengths - self.wavelengths[0]) / self.wavelength_step
            query_indices = dr.uint32_array_t(mi.Spectrum)(dr.floor(normalized_wavelengths))

            # Get fractional part of the indices
            lerp_factor = normalized_wavelengths - query_indices

            for i in range(len(si.wavelengths)):
                idx = query_indices[i]

                # Deactivate wrong indices, (no need to check "< 0" since they are unsigned)
                mask = active & (idx < len(self.wavelengths))

                res[i] = dr.lerp(self.render_channel(idx, cos_theta, cos_gamma, mask),
                                 self.render_channel(dr.minimum(idx + 1, 10), cos_theta, cos_gamma, mask),
                                 lerp_factor[i])

        return res & active


    def sample_direction(self, it, sample, active=True) -> (mi.DirectionSample3f, mi.Spectrum):

        # Sample a gaussian
        glb_idx, sample[0] = self.gaussian_dist.sample_reuse(sample[0], active)
        gaussian = dr.gather(mi.ArrayXf, self.gaussians, glb_idx, active, shape=(NB_GAUSSIAN_PARAMS, 1))
        local_direction = sample_gaussian(sample, gaussian, self.sun_phi)

        # Get PDF
        inv_st = inv_sin_theta(local_direction)
        pdf = tgmm_pdf(self.gaussians, local_direction, self.sun_phi, active) * inv_st
        active &= pdf > 0

        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2 * radius

        direction = dr.normalize(self.to_world @ local_direction)
        ds = mi.DirectionSample3f(
            p=dr.fma(direction, dist, it.p),
            n=-direction,
            uv=sample,
            time=it.time,
            pdf=pdf & active,
            delta=mi.Bool(False),
            d=direction,
            dist=dist,
            emitter=mi.EmitterPtr(self))

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = it.wavelengths

        return ds, self.eval(si, active) / pdf

    def pdf_direction(self, it, ds, active=True):
        local_direction = dr.normalize(self.to_world.inverse() @ ds.d)
        inv_st = inv_sin_theta(local_direction)

        return tgmm_pdf(self.gaussians, local_direction, self.sun_phi, active) * inv_st

    def is_environment(self):
        return True

mi.register_emitter("sunsky", SunskyEmitter)
