import sys
sys.path.insert(0, "build/python")

import itertools

import drjit as dr
import mitsuba as mi

def render_and_compare(ref_path, params: tuple[float]):
    from spectral_render_test import eval_full_spec

    global sp_sun, cp_sun
    sun_theta = HALF_PI - params[0]
    st, ct = dr.sincos(sun_theta)

    render_res = (1024//2, 1024)

    sky = mi.load_dict({
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sun_scale": 0.0,
        "turbidity": params[1],
        "albedo": params[2],
        "albedo_test": params[2]
    })

    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.two_pi, render_res[1]),
        dr.linspace(mi.Float, dr.pi, 0, render_res[0]))
    sp, cp = dr.sincos(phis)
    st, ct = dr.sincos(thetas)

    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(cp*st, sp*st, ct)

    rendered_scene = mi.TensorXf(dr.ravel(sky.eval(si)), (*render_res, 3)) if mi.is_rgb \
                else eval_full_spec(sky, si, wavelengths, render_res)
    reference_scene = mi.TensorXf(mi.Bitmap(ref_path))

    rtol = 0.005 if mi.is_rgb else 0.0354
    rel_err = dr.mean(dr.abs(rendered_scene - reference_scene) / (dr.abs(reference_scene) + 0.001))
    if rel_err > rtol:
        print(f"Fail when rendering {params=}")
        print("Reference is ", ref_path)
        mi.util.write_bitmap("sunsky-testing/res/renders/fail.exr", rendered_scene)
        exit(1)

    dr.print("Pass when rendering t={t}, a={a}, eta={eta}", t= params[1], a=params[2], eta=params[0])


if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")

    HALF_PI = dr.pi/2
    sun_phi = 0
    sp_sun, cp_sun = dr.sincos(sun_phi)

    start, end = 360, 830
    step = (end - start) / 10
    wavelengths = [start + step/2 + i*step for i in range(10)]

    def make_range_idx(nb, a, b):
        return [(i/(nb-1)) * (b - a) + a for i in range(nb)]

    etas = make_range_idx(5, 0, HALF_PI)
    turbs = make_range_idx(7, 1, 10)
    albedos = make_range_idx(3, 0, 1)

    render_type = "rgb" if mi.is_rgb else "spec"

    for (eta, turb, albedo) in itertools.product(etas, turbs, albedos):
        path = f"../renders/{render_type}/sky_{render_type}_eta{eta:.3f}_t{turb:.3f}_a{albedo:.3f}.exr"
        render_and_compare(path, (eta, turb, albedo))
