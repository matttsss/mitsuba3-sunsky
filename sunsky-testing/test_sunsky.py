import sys
sys.path.insert(0, "build/python")

import numpy as np
import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_spectral")

from sunsky_data import get_params, get_rad
from test_plugin import ConstantEmitter
from sunsky_plugin import SunskyEmitter

mi.register_emitter("sunsky_emitter", lambda props: SunskyEmitter(props))
mi.register_emitter("constant_emitter", lambda props: ConstantEmitter(props))


def test_mean_radiance_data():
    dataset_rad: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v1_spec_rad.bin")

    # Control points for mean radiance at a = 0, t = 2, lbda = 0
    expected = mi.Float(9.160628e-004, 2.599956e-004, 5.466998e-003, -1.503537e-002, 7.200167e-002, 5.387713e-002)
    dr.assert_true(dr.allclose(expected, dataset_rad[0, 1, 0]), "Incorrect values for mean radiance (a=0, t=2, lbda=0)")

    # Control points for mean radiance at a = 1, t = 6, lbda = 5
    expected = mi.Float(1.245635e-002, 2.874175e-002, -6.384005e-002, 2.429023e-001, 2.428387e-001, 2.418906e-001,)
    dr.assert_true(dr.allclose(expected, dataset_rad[1, 5, 5]), "Incorrect values for mean radiance (a=1, t=6, lbda=5)")

def test_radiance_data():
    dataset: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v1_spec.bin")

    # Control points for radiance at a = 0, t = 2, lbda = 0
    expected = mi.Float(-1.298333e+001, -3.775577e+000, -5.173531e+000, 5.316518e+000, -2.572615e-002, 1.516601e-001,
                        -8.297168e-006, 1.669649e+000, 9.000495e-001, -1.402639e+001, -3.787558e+000, 7.611941e-002,
                        2.521881e-001, -5.859973e-002, 1.753711e-001, 4.670097e-005, 1.459275e+000, 8.998629e-001,
                        -2.190256e+000, -3.575495e+000, -4.930996e-001, 4.826321e-002, -6.797145e-002, 3.425922e-002,
                        -3.512550e-004, 1.978419e+000, 8.866517e-001, -2.415991e+000, -1.453294e+000, 2.170671e-001,
                        1.341284e-001, -1.926330e-001, 1.059103e-001, 1.360739e-003, 1.587725e+000, 9.821154e-001,
                        -5.254592e-001, -8.181026e-001, 7.535702e-001, -3.323364e-002, 4.503149e-001, 5.778285e-001,
                        -4.089673e-003, 3.335089e-001, 6.827164e-001, -1.280108e+000, -1.013716e+000, 5.577676e-001,
                        9.539205e-004, -4.934956e+000, 2.642883e-001, 1.005169e-002, 9.265844e-001, 4.999698e-001)

    dr.assert_true(dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[0, 1, 0]), "C")),
                   "Incorrect values for radiance (a=0, t=2, lbda=0)")

    # Control points for radiance at a = 1, t = 6, lbda = 5
    expected = mi.Float(-1.330626e+000, -4.272516e-001, -1.317682e+000, 1.650847e+000, -1.192771e-001, 4.904191e-001,
                        4.074827e-002, 3.015846e+000, 5.271835e-001, -1.711989e+000, -8.644776e-001, 4.057135e-001,
                        9.037139e-001, -3.100084e-001, 6.317697e-002, 1.065625e-001, 9.226240e-002, 3.022474e-001,
                        -8.465894e-001, 1.652715e-001, -2.532361e-001, -2.422693e+000, 3.144841e-001, 1.839347e+000,
                        -2.818162e-001, 7.856667e+000, 1.387977e+000, -1.192114e+000, -3.830569e-001, 5.124751e-001,
                        7.280034e+000, -2.610477e+000, -1.832768e+000, 9.101904e-001, -3.349116e+000, -7.313079e-002,
                        -1.011026e+000, -1.061217e-001, 1.357854e+000, -1.496195e+001, -2.180975e+000, 2.484329e+000,
                        -3.239225e-001, 3.899425e+000, 1.179264e+000, -1.106228e+000, -1.927917e-001, 1.179701e+000,
                        2.379834e+001, -4.870211e+000, -1.290713e+000, 2.854422e-001, 2.078973e+000, 5.128625e-001)

    dr.assert_true(dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[1, 5, 5]), "C")),
                   "Incorrect values for radiance (a=1, t=6, lbda=5)")

def test_compute():
    dataset_rad: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v1_rgb_rad.bin")
    dataset: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v1_rgb.bin")

    # Set test parameters
    W, H = (256*4, 256)

    t, a = 6, 0.5
    solar_elevation = (dr.pi / 2) * 15 / 100

    # Compute coefficients
    params = get_params(dataset, solar_elevation, t, a)
    mean_radiance = get_params(dataset_rad, solar_elevation, t, a)

    dr.assert_true(params.shape == (3, 9), "Parameters are not of the right shape")
    dr.assert_true(mean_radiance.shape == (3,), "Mean radiance is not of the right shape")


    # Compute angles for testing
    phi, thetas = dr.meshgrid(
        dr.linspace(mi.Float, -dr.pi, dr.pi, W),
        dr.linspace(mi.Float, dr.pi / 2, 0, H)
    )

    st, ct = dr.sincos(thetas)
    sp, cp = dr.sincos(phi)
    view_dir = mi.Vector3f(cp * st, sp * st, ct)

    sun_dir = mi.Vector3f(dr.sin(solar_elevation), 0, dr.cos(solar_elevation))

    gammas = dr.safe_acos(dr.dot(view_dir, sun_dir))

    res = [get_rad(params[i], thetas, gammas) * mean_radiance[i] for i in range(params.shape[0])]
    res = np.stack([dr.reshape(mi.TensorXf, channel, (H, W)) for channel in res]).transpose((1,2,0))

    #mi.util.write_bitmap("sunsky-testing/res/latlong_test.exr", res)






spectrum_dicts = {
    'd65': {
        "type": "d65",
    },
    'regular': {
        "type": "regular",
        "wavelength_min": 500,
        "wavelength_max": 600,
        "values": "1, 2"
    }
}

def create_emitter_and_spectrum(s_key='d65'):
    emitter = mi.load_dict({
        "type" : "constant_emitter",
        "radiance" : spectrum_dicts[s_key]
    })
    spectrum = mi.load_dict(spectrum_dicts[s_key])
    expanded = spectrum.expand()
    if len(expanded) == 1:
        spectrum = expanded[0]

    return emitter, spectrum

def chi2_test(variants_vec_spectral, spectrum_key):
    sse_dict = {
        'type' : 'constant_emitter',
        'radiance' : spectrum_dicts[spectrum_key]
    }

    sample_func, pdf_func = mi.chi2.EmitterAdapter("constant_emitter", sse_dict)
    chi2 = mi.chi2.ChiSquareTest(
        domain = mi.chi2.SphericalDomain(),
        sample_func = sample_func,
        pdf_func = pdf_func
    )

    assert chi2.run()





if __name__ == "__main__":
    chi2_test(None, 'd65')

    mi.write_sky_model_data_v1("sunsky-testing/res/ssm_dataset")
    test_mean_radiance_data()
    test_radiance_data()
    test_compute()

