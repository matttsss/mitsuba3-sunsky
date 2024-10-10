import drjit as dr
import mitsuba as mi

from sunsky_data import get_params, get_rad


class SunskyEmitter(mi.Emitter):

    def __init__(self, props):
        super().__init__(props)

        self.m_bsphere = mi.BoundingSphere3f(mi.Point3f(0), 1)
        self.m_surface_area = 4.0 * dr.pi

        self.m_sun_dir = dr.normalize(props.get("sun_dir", mi.Vector3f(dr.sin(0.5 * dr.pi * 70/100), 0, dr.cos(0.5 * dr.pi * 70/100))))
        self.m_albedo = props.get("albedo", 0.5)
        self.m_turb = props.get("turbidity", 6)

        dataset_name = props.get("dataset_name", "sunsky-testing/res/ssm_dataset_v1_rgb")
        _, database = mi.array_from_file(dataset_name + ".bin")
        _, database_rad = mi.array_from_file(dataset_name + "_rad.bin")

        if mi.is_spectral:
            self.wavelengths = mi.Spectrum([320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720])
            self.wavelength_step = 40

        # TODO get to_world from props
        self.m_up_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
        sun_elevation = dr.pi/2 - dr.acos(self.m_up_frame.cos_theta(self.m_sun_dir))

        self.m_params = get_params(database, self.m_turb, self.m_albedo, sun_elevation)
        self.m_rad = get_params(database_rad, self.m_turb, self.m_albedo, sun_elevation)

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


    def render_channel(self, i, cos_theta, cos_gamma):
        coefs = dr.gather(mi.Float, self.m_params, i*9 + dr.arange(mi.UInt32, 9))
        gamma = dr.acos(cos_gamma)
        cos_gamma_sqr = dr.square(cos_gamma)

        c1 = 1 + coefs[0] * dr.exp(coefs[1] / (cos_theta + 0.01))
        chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(coefs[7]) - 2 * coefs[7] * cos_gamma, 1.5)
        c2 = coefs[2] + coefs[3] * dr.exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[8] * dr.sqrt(cos_theta)


        return dr.select(cos_theta >= 0, c1 * c2 * self.m_rad[i], 0)


    @dr.syntax
    def eval(self, si, active=True):
        world_wi = dr.normalize(si.to_world(si.wi))
        cos_theta = self.m_up_frame.cos_theta(world_wi)
        cos_gamma = dr.dot(self.m_sun_dir, world_wi)

        res = dr.zeros(mi.Spectrum)

        if mi.is_rgb: # TODO optimize RGB with active
            res = dr.zeros(mi.Spectrum)
            res[0] = self.render_channel(0, cos_theta, cos_gamma)
            res[1] = self.render_channel(1, cos_theta, cos_gamma)
            res[2] = self.render_channel(2, cos_theta, cos_gamma)

        else:
            normalized_wavelengths = (si.wavelengths - self.wavelengths[0]) / self.wavelength_step
            query_indices = mi.Int(dr.floor(normalized_wavelengths))

            lerp_factor = (normalized_wavelengths - query_indices)

            i = 0
            while active and i < len(si.wavelengths):
                query_idx = query_indices[i]
                if query_idx < 0 or query_idx >= len(self.wavelengths):
                    res[i] = 0
                else:
                    res[i] = dr.lerp(self.render_channel(query_idx, cos_theta, cos_gamma),
                                     self.render_channel(dr.minimum(query_idx + 1, 10), cos_theta, cos_gamma),
                                     lerp_factor[i])
                i += 1

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

        # TODO, is the size of the array returned by sample_shifted allways the same as the nb of wavelengths?
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
