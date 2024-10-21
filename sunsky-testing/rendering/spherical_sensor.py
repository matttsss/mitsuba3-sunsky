from typing import override

import drjit as dr
import mitsuba as mi


class SphericalSensor(mi.Sensor):

    @override
    def __init__(self, props):
        super().__init__(props)

        self.transform = props.get("to_world", mi.Transform4f(1.0))
        self.up = self.transform.transform_affine(mi.Vector3f(0, 0, 1))

    #@override
    #def eval(self, si, active=True):
    #    return mi.Spectrum(1.0) & active

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

    #@override
    #def sample_direction(self, it, sample, active = True):
    #    ds = dr.zeros(mi.DirectionSample3f)
#
    #    # FIXME normalised UVs?
    #    ds.uv = sample * super().film()
    #    ds.time = it.time
#
#
    #    sp, cp = dr.sincos(sample.x * dr.two_pi)
    #    st, ct = dr.sincos(sample.y * dr.pi)
#
    #    ds.d = self.transform @ mi.Vector3f(sp * st, ct, -cp * st)
    #    ds.pdf = 1 / (dr.two_pi * dr.maximum(st, dr.epsilon(mi.Float)))
#
    #    return ds, mi.Spectrum(1.0)

    #@override
    #def sample_position(self, time, sample, active = True):
    #    pos_sample = dr.zeros(mi.PositionSample3f)
    #    pos_sample.p = self.transform.transform_affine(mi.Point3f(0.0))
    #    pos_sample.n = mi.Normal3f(0.0)
    #    pos_sample.time = time
    #    pos_sample.pdf = 1.0
    #    pos_sample.delta = True
    #    return pos_sample

    @override
    def sample_ray(self, time, sample1, sample2, sample3, active = True):
        st, ct = dr.sincos(sample2.y * dr.pi)
        sp, cp = dr.sincos(sample2.x * dr.two_pi)

        ray = mi.Ray3f(mi.Point3f(0.0), mi.Vector3f(sp*st, ct, -cp * st), time)

        if dr.hint(mi.is_spectral, mode="scalar"):
            ray.wavelengths = mi.Spectrum(mi.sample_shifted(sample1))

        return self.transform.transform_affine(ray), mi.Spectrum(1.0)


mi.register_sensor("spherical", SphericalSensor)
