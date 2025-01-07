from typing import override

import drjit as dr
import mitsuba as mi

class SphericalSensor(mi.Sensor):

    @override
    def __init__(self, props):
        super().__init__(props)

        self.transform = props.get("to_world", mi.Transform4f(1.0))
        self.up = self.transform.transform_affine(mi.Vector3f(0, 0, 1))

    @override
    def sample_ray(self, time, sample1, sample2, sample3, active = True):
        res = mi.Spectrum(1.0)
        ray = mi.Ray3f(mi.Point3f(0.0), mi.warp.square_to_uniform_sphere(sample2), time)

        if dr.hint(mi.is_spectral, mode="scalar"):
            si = dr.zeros(mi.SurfaceInteraction3f)
            ray.wavelengths, res = super().sample_wavelengths(si, sample1, active)
            #ray.wavelengths = mi.sample_shifted(sample1) * (mi.MI_CIE_MAX - mi.MI_CIE_MIN) + mi.MI_CIE_MIN

        return self.transform.transform_affine(ray), res

    def to_string(self):
        return f"SphericalSensor[\nfilm: {super().film()}\n]"


mi.register_sensor("spherical", SphericalSensor)
