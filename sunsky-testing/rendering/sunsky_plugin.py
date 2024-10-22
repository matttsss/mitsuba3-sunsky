import drjit as dr
import mitsuba as mi

from .sunsky_data import get_params

def get_class(name):
    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    return value

def get_module(class_):
    return get_class(class_.__module__)

ArrayXf = get_module(mi.Float).ArrayXf

class SunskyEmitter(mi.Emitter):

    def __init__(self, props):
        super().__init__(props)

        self.m_bsphere = mi.BoundingSphere3f(mi.Point3f(0), 1)
        self.m_surface_area = 4.0 * dr.pi

        self.m_albedo = props.get("albedo", 0.15)
        self.m_turb = props.get("turbidity", 3)

        sun_elevation = 0.5 * dr.pi * (2/100)
        self.m_up = props.get("to_world", mi.Transform4f(1)) @ mi.Vector3f(0, 0, 1)
        self.m_sun_dir = dr.normalize(props.get("sun_direction", mi.Vector3f(dr.cos(sun_elevation), 0, dr.sin(sun_elevation))))
        sun_elevation = dr.pi/2 - dr.acos(dr.dot(self.m_sun_dir, self.m_up))


        if mi.is_spectral:
            dataset_name = props.get("dataset_name", "data/ssm_dataset_v2_spec")

            self.wavelengths = [320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720]
            self.wavelength_step = 40

        elif mi.is_rgb:
            dataset_name = props.get("dataset_name", "data/ssm_dataset_v2_rgb")

        _, database = mi.array_from_file(dataset_name + ".bin")
        _, database_rad = mi.array_from_file(dataset_name + "_rad.bin")
        self.m_params = get_params(database, self.m_turb, self.m_albedo, sun_elevation)
        self.m_rad = get_params(database_rad, self.m_turb, self.m_albedo, sun_elevation)

        dr.eval(self.m_params, self.m_rad)

        self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying

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
        coefs = dr.gather(ArrayXf, self.m_params, idx, active, shape=(9, 1))

        gamma = dr.acos(cos_gamma)
        cos_gamma_sqr = dr.square(cos_gamma)

        c1 = 1 + coefs[0] * dr.exp(coefs[1] / (cos_theta + 0.01))
        chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5)
        c2 = coefs[2] + coefs[3] * dr.exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr.sqrt(cos_theta)

        return c1 * c2 * dr.gather(mi.Float, self.m_rad, idx, active) & (cos_theta >= 0)

    @dr.syntax
    def eval(self, si, active=True):
        cos_theta = dr.dot(self.m_up, -si.wi)
        cos_gamma = dr.dot(self.m_sun_dir, -si.wi)

        active &= cos_theta >= 0

        res = dr.zeros(mi.Spectrum)
        if dr.hint(mi.is_rgb, mode="scalar"):
            res[0] = self.render_channel(0, cos_theta, cos_gamma, active)
            res[1] = self.render_channel(1, cos_theta, cos_gamma, active)
            res[2] = self.render_channel(2, cos_theta, cos_gamma, active)

            res *= mi.MI_CIE_D65_NORMALIZATION

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
        v0 = mi.warp.square_to_uniform_sphere(sample_2)
        ray_orig = mi.Point3f(dr.fma(v0, self.m_bsphere.radius,
                                     self.m_bsphere.center))

        # Direction sampling
        v1 = mi.warp.square_to_cosine_hemisphere(sample_3)
        ray_dir = mi.Frame3f(-v0).to_world(v1)

        # Spectral sampling
        wavelengths, weights = self.sample_wavelengths(
            mi.SurfaceInteraction3f(), wavelength_sample, active)

        weights *= self.m_surface_area * dr.pi

        return mi.Ray3f(ray_orig, ray_dir, time, wavelengths), mi.depolarizer(weights)



    def sample_direction(self, it, sample, active=True) -> (mi.DirectionSample3f, mi.Spectrum):
        direction = mi.warp.square_to_uniform_sphere(sample)

        # TODO why enlarge radius? bc of volumetric rendering?
        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2*radius

        # TODO what is delta arg
        ds = mi.DirectionSample3f(
            p = dr.fma(direction, dist, it.p),
            n = -direction,
            uv = sample,
            time = it.time,
            pdf = mi.warp.square_to_uniform_sphere_pdf(direction),
            delta = mi.Bool(False),
            d = direction,
            dist = dist,
            emitter = mi.EmitterPtr(self))

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = it.wavelengths

        return ds, self.eval(si, active) / ds.pdf

    def pdf_direction(self, it, ds, active=True):
        return mi.warp.square_to_uniform_sphere_pdf(ds.d)


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

    def sample_position(self, ref, ds, active = True):
        dr.assert_true(False, "Sample position not implemented")


    def traverse(self, callback):
        callback.put_parameter('sun_dir', self.m_sun_dir)
        callback.put_parameter('albedo', self.m_albedo)
        callback.put_parameter('turbidity', self.m_turb)

    def is_environment(self):
        return True

mi.register_emitter("sunsky", SunskyEmitter)
