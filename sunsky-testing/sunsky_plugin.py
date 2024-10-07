import drjit as dr
import mitsuba as mi

from sunsky_data import get_params, get_rad


class SunskyEmitter(mi.Emitter):

    def __init__(self, props):
        super().__init__(props)

        self.m_bsphere = mi.BoundingSphere3f(mi.Point3f(0), mi.Float(1))
        self.m_surface_area = 4.0 * dr.pi

        #self.m_radiance: mi.Texture = props.get("radiance")
        #dr.assert_false(self.m_radiance.is_spatially_varying())

        self.m_sun_dir = dr.normalize(props.get("sun_dir", mi.Vector3f(dr.sin(dr.pi/5), 0, dr.cos(dr.pi/5))))
        self.m_albedo = props.get("albedo", 0.5)
        self.m_turb = props.get("turbidity", 6)

        dataset_name = props.get("dataset_name", "sunsky-testing/res/ssm_dataset_v1_rgb")
        _, database = mi.tensor_from_file(dataset_name + ".bin")
        _, database_rad = mi.tensor_from_file(dataset_name + "_rad.bin")

        if mi.is_spectral:
            self.wavelengths = mi.Spectrum([320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720])
            self.wavelength_step = 40


        self.m_up_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
        sun_elevation = dr.pi/2 - dr.acos(self.m_up_frame.cos_theta(self.m_sun_dir))

        self.m_params = get_params(database, self.m_turb, self.m_albedo, sun_elevation)
        self.m_rad: mi.Float = get_params(database_rad, self.m_turb, self.m_albedo, sun_elevation)

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

    @dr.syntax
    def eval(self, si, active=True):
        gamma = dr.acos(dr.dot(si.wi, self.m_sun_dir))
        theta = dr.acos(self.m_up_frame.cos_theta(si.wi))

        if mi.is_rgb:
            res = dr.zeros(mi.Spectrum, len(gamma))
            res.x = get_rad(self.m_params[0], theta, gamma) * self.m_rad[0]
            res.y = get_rad(self.m_params[1], theta, gamma) * self.m_rad[1]
            res.z = get_rad(self.m_params[2], theta, gamma) * self.m_rad[2]
            return res

        else:
            res = dr.zeros(mi.Spectrum, len(gamma))

            i = 0
            while active and i < len(si.wavelengths):
                query_val = si.wavelengths[i]
                query_idx = mi.Int((query_val - self.wavelengths[0]) / self.wavelength_step)
                if query_idx < 0 or query_idx > len(self.wavelengths) - 1:
                    res[i] = 0
                else:
                    coefs = dr.gather(mi.Float, self.m_params, query_idx * 9 + dr.arange(mi.Float, 9))
                    temp = get_rad(coefs, theta, gamma) * self.m_rad[query_idx]

                i += 1


        # TODO construct spectrum
        return mi.depolarizer(self.m_radiance.eval(si, active))

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

        return ds, mi.depolarizer(self.m_radiance.eval(si, active)) / ds.pdf

    def pdf_direction(self, it, ds, active=True):
        return mi.warp.square_to_uniform_sphere_pdf(ds.d)


    def eval_direction(self, it, ds, active=True):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = it.wavelengths
        return mi.depolarizer(self.m_radiance.eval(si, active))

    def sample_wavelengths(self, si, sample, active = True):
        return self.m_radiance.sample_spectrum(si, mi.sample_shifted(sample), active)

    def sample_position(self, ref, ds, active = True):
        dr.assert_true(False, "Sample position not implemented")


    def traverse(self, callback):
        callback.put_parameter('sun_dir', self.m_sun_dir)
        callback.put_parameter('albedo', self.m_albedo)
        callback.put_parameter('turbidity', self.m_turb)

    def is_environment(self):
        return True
