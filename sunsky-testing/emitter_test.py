import sys; sys.path.insert(0, "build/python")

import itertools

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")

from rendering.sunsky_plugin import SunskyEmitter
from rendering.spherical_sensor import SphericalSensor

HALF_PI = dr.pi/2
sun_phi = 0
sp_sun, cp_sun = dr.sincos(sun_phi)

def render_and_compare(ref_path, params: tuple[float]):
    global sp_sun, cp_sun
    sun_theta = HALF_PI - params[0]
    st, ct = dr.sincos(sun_theta)

    sky = mi.load_dict({
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "turbidity": params[1],
        "albedo": params[2]
    })

    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.two_pi, 1024),
        dr.linspace(mi.Float, dr.pi, 0, 1024//2))
    sp, cp = dr.sincos(phis)
    st, ct = dr.sincos(thetas)

    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(cp*st, sp*st, ct)
    rendered_scene = sky.eval(si)

    rendered_scene = dr.reshape(mi.TensorXf, rendered_scene, (1024//2, 1024, 3))
    reference_scene = mi.Bitmap(ref_path)

    rtol = 0.11
    atol = rtol/100
    if not dr.allclose(rendered_scene, mi.TensorXf(reference_scene), rtol=rtol, atol=atol):
        print(f"Fail when rendering {params=}")
        print("Reference is ", ref_path)
        mi.util.write_bitmap("sunsky-testing/res/renders/fail.exr", rendered_scene)
        exit(1)



def tests():
    def make_range_idx(nb, a, b):
        return [(i/(nb-1)) * (b - a) + a for i in range(nb)]

    etas = make_range_idx(5, 0, HALF_PI)
    turbs = make_range_idx(7, 1, 10)
    albedos = make_range_idx(3, 0, 1)


    for (eta, turb, albedo) in itertools.product(etas, turbs, albedos):
        path = f"../renders/sky_rgb_eta{eta:.3f}_t{turb:.3f}_a{albedo:.3f}.exr"
        render_and_compare(path, (eta, turb, albedo))


tests()