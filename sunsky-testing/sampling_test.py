import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

def get_dict(t, a, eta, sky_scale=1, sun_scale=0, phi=-4*dr.pi/5):
    sp_sun, cp_sun = dr.sincos(phi)
    st, ct = dr.sincos(dr.pi/2 - eta)

    return {
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sky_scale": sky_scale,
        "sun_scale": sun_scale,
        "turbidity": t,
        "albedo": a
    }

def check_sampling():
    t, a = 6, 0.5
    eta = dr.deg2rad(50.2)
    plugin_dict = get_dict(t, a, eta)

    sky_emitter = mi.load_dict(plugin_dict)
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


    sample_func, pdf_func = mi.chi2.EmitterAdapter("sunsky", plugin_dict)
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


def plot_sun_sky_weight():
    import numpy as np
    import matplotlib.pyplot as plt

    res = (100, 100)
    turbs = np.linspace(1, 10, res[0])
    etas  = np.linspace(0, 90, res[1]) * np.pi / 180

    weights = np.zeros(res)
    for i in range(res[0]):
        for j in range(res[1]):
            plugin = mi.load_dict(get_dict(turbs[i], 0.5, etas[j], 1, 1))
            weights[i, j] = plugin.pdf_direction(mi.Interaction3f(), mi.DirectionSample3f(), True)[0]

    print(np.mean(weights))
    etas = etas * 180 / np.pi
    plt.imshow(weights, extent=(etas[0], etas[-1], turbs[0], turbs[-1]), origin="lower", aspect="auto")
    plt.colorbar()
    plt.ylabel("Turbidity")
    plt.xlabel("Elevation angle (°)")
    plt.title("Probability to sample the sky over the sun")
    plt.show()


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

    plot_sun_sky_weight()
    #check_sampling()

