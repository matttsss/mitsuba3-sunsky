import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")

from rendering.sunsky_plugin import SunskyEmitter

def test_sampling():
    t, a = 6, 0.5
    eta = dr.deg2rad(50.2)
    phi_sun = -4*dr.pi/5

    #t, a = 3, 0.5
    #eta = dr.deg2rad(74)
    #phi_sun = -4*dr.pi/5

    sp_sun, cp_sun = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)

    plugin_name = "sunsky"

    sky = {
        "type": plugin_name,
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sun_scale": 0.0,
        "turbidity": t,
        "albedo": a
    }
    sky_emitter = mi.load_dict(sky)
    it = dr.zeros(mi.Interaction3f)

    nb_samples = 250_000_000

    rng = mi.PCG32(size=2*nb_samples)
    samples = rng.next_float32()
    samples = dr.reshape(mi.Point2f, samples, shape=(nb_samples, 2))

    ds = sky_emitter.sample_direction(it, samples)[0]

    problematic_indices = dr.compress((ds.d.z < 0))
    count = len(problematic_indices)
    problematic_quantiles = dr.gather(mi.Point2f, samples, problematic_indices)
    problematic_samples = dr.gather(mi.Vector3f, ds.d, problematic_indices)


    dr.print("Nb in lower hemisphere: {count}", count=count)
    dr.print("Problematic quantiles: {quantiles}\n", quantiles=problematic_quantiles)
    dr.print("Problematic samples: {samples}\n", samples=problematic_samples)


    sample_func, pdf_func = mi.chi2.EmitterAdapter(plugin_name, sky)
    test = mi.chi2.ChiSquareTest(
        domain=mi.chi2.SphericalDomain(),
        pdf_func= pdf_func,
        sample_func= sample_func,
        sample_dim=2,
        sample_count=nb_samples,
        res=216,
        ires=32
    )

    dr.print("Chi2 passes: {chi2_pass}", chi2_pass=test.run())


if __name__ == "__main__":
    test_sampling()

