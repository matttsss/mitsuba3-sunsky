import drjit as dr
import mitsuba as mi

from .sunsky_data import (get_params, get_tgmm_table, GAUSSIAN_WEIGHT_IDX,
                          NB_GAUSSIAN_PARAMS, sample_gaussian, tgmm_pdf)

SIN_OFFSET = 0.01


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
        self.m_up = self.to_world @ mi.Vector3f(0, 0, 1)
        self.m_sun_dir = dr.normalize(props.get("sun_direction"))

        self.frame = mi.Frame3f(self.m_up)
        self.sun_phi = dr.acos(self.frame.cos_phi(self.m_sun_dir))
        sun_eta = dr.pi / 2 - dr.acos(self.frame.cos_theta(self.m_sun_dir))

        # Get luminance parameters
        _, database = mi.array_from_file_d(dataset_name + ".bin")
        _, database_rad = mi.array_from_file_d(dataset_name + "_rad.bin")
        self.m_params = get_params(database, turb, albedo, sun_eta)
        self.m_rad = get_params(database_rad, turb, albedo, sun_eta)

        # Get sampling parameters
        _, tgmm_tables = mi.array_from_file_f("sunsky-testing/res/datasets/tgmm_tables.bin")
        self.tgmm_table = get_tgmm_table(tgmm_tables, turb, sun_eta)
        self.gaussian_dist = mi.DiscreteDistribution(dr.gather(mi.Float, self.tgmm_table, GAUSSIAN_WEIGHT_IDX))
        self.gaussian_dist.update()

        self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying

        dr.eval(self.m_params, self.m_rad, self.tgmm_table, self.gaussian_dist)

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
        cos_theta = self.frame.cos_theta(-si.wi)
        cos_gamma = dr.dot(self.m_sun_dir, -si.wi)

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

        return res


    def sample_ray(self, time, wavelength_sample, sample_2, sample_3, active=True):
        # Spacial sampling
        gaussian_idx, sample_2[0] = self.gaussian_dist.sample_reuse(sample_2[0], active)
        gaussian = dr.gather(mi.ArrayXf, self.tgmm_table, gaussian_idx, active, shape=(NB_GAUSSIAN_PARAMS, 1))

        v0 = sample_gaussian(sample_2, gaussian, self.sun_phi)
        v0 = dr.normalize(self.to_world @ v0)
        ray_orig = mi.Point3f(dr.fma(v0, self.m_bsphere.radius,
                                     self.m_bsphere.center))

        # Direction sampling
        v1 = mi.warp.square_to_cosine_hemisphere(sample_3)
        ray_dir = mi.Frame3f(-v0).to_world(v1)

        # Spectral sampling
        wavelengths, weights = self.sample_wavelengths(
            mi.SurfaceInteraction3f(), wavelength_sample, active)

        weights *= dr.maximum(self.frame.sin_theta(ray_dir), SIN_OFFSET)
        weights /= tgmm_pdf(self.tgmm_table, v0, self.sun_phi, active)

        return mi.Ray3f(ray_orig, ray_dir, time, wavelengths) & active, mi.depolarizer(weights) & active

    def sample_direction(self, it, sample, active=True) -> (mi.DirectionSample3f, mi.Spectrum):
        gaussian_idx, sample[0] = self.gaussian_dist.sample_reuse(sample[0], active)
        gaussian = dr.gather(mi.ArrayXf, self.tgmm_table, gaussian_idx, active, shape=(NB_GAUSSIAN_PARAMS, 1))

        local_direction = sample_gaussian(sample, gaussian, self.sun_phi)
        direction = dr.normalize(self.to_world @ local_direction)
        sin_theta = dr.maximum(self.frame.sin_theta(direction), SIN_OFFSET)

        pdf = tgmm_pdf(self.tgmm_table, local_direction, self.sun_phi, active) / sin_theta

        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2 * radius

        ds = mi.DirectionSample3f(
            p=dr.fma(direction, dist, it.p),
            n=-direction,
            uv=sample,
            time=it.time,
            pdf=pdf,
            delta=mi.Bool(False),
            d=direction,
            dist=dist,
            emitter=mi.EmitterPtr(self))

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = it.wavelengths

        return ds, self.eval(si, active) / pdf

    def pdf_direction(self, it, ds, active=True):
        local_direction = dr.normalize(self.to_world.inverse() @ ds.d)
        sin_theta = self.frame.sin_theta(ds.d)
        return tgmm_pdf(self.tgmm_table, local_direction, self.sun_phi, active) / (dr.maximum(sin_theta, SIN_OFFSET))

    def eval_direction(self, it, ds, active=True):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = it.wavelengths
        return self.eval(si, active)

    def sample_wavelengths(self, si, sample, active = True):
        min_lbda = dr.maximum(self.wavelengths[0], mi.MI_CIE_MIN)
        max_lbda = dr.minimum(self.wavelengths[-1], mi.MI_CIE_MAX)
        inv_pdf = max_lbda - min_lbda

        si.wavelengths = mi.sample_shifted(sample) * inv_pdf + min_lbda
        return si.wavelengths, inv_pdf * self.eval(si, active)

    def is_environment(self):
        return True

mi.register_emitter("sunsky", SunskyEmitter)
