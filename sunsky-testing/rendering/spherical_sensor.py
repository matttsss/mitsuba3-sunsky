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
    def eval_direction(self, it, ds, active = True):
        return mi.Spectrum(self.pdf_direction(it, ds, active))

    @override
    def pdf_direction(self, it, ds, active = True):
        st = dr.safe_sqrt(1 - dr.dot(ds.d, self.up)**2)
        return (1 / (dr.two_pi * dr.maximum(st, dr.epsilon(mi.Float)))) & active

    @override
    def pdf_position(self, ps, active = True):
        return 0.0

    @override
    def sample_ray(self, time, sample1, sample2, sample3, active = True):
        st, ct = dr.sincos(sample2.y * dr.pi)
        sp, cp = dr.sincos(sample2.x * dr.two_pi)

        ray = mi.Ray3f(mi.Point3f(0.0), mi.Vector3f(cp*st, sp*st, ct), time)

        if dr.hint(mi.is_spectral, mode="scalar"):
            ray.wavelengths = mi.Spectrum(mi.sample_shifted(sample1))

        return self.transform.transform_affine(ray), mi.Spectrum(1.0)


mi.register_sensor("spherical", SphericalSensor)
