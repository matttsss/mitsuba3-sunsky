import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant("llvm_rgb")

from helpers import get_north_hemisphere_rays
from sunsky_data import get_params, get_rad, get_sun


def test_mean_radiance_data():
    dr.print("Testing mean radiance values")

    rad_shape, dataset_rad = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec_rad.bin")
    dataset = mi.TensorXf(dataset_rad, tuple(rad_shape))

    # Control points for mean radiance at a = 0, t = 2, lbda = 0
    expected = mi.Float(9.160628e-004, 2.599956e-004, 5.466998e-003, -1.503537e-002, 7.200167e-002, 5.387713e-002)
    dr.assert_true(dr.allclose(expected, dataset[0, 1, ::, 0]), "Incorrect values for mean radiance (a=0, t=2, lbda=0)")

    # Control points for mean radiance at a = 1, t = 6, lbda = 5
    expected = mi.Float(1.245635e-002, 2.874175e-002, -6.384005e-002, 2.429023e-001, 2.428387e-001, 2.418906e-001,)
    dr.assert_true(dr.allclose(expected, dataset[1, 5, ::, 5]), "Incorrect values for mean radiance (a=1, t=6, lbda=5)")

def test_radiance_data():
    dr.print("Testing radiance values")

    shape, dataset = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec.bin")
    dataset = mi.TensorXf(dataset, tuple(shape))

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

    dr.assert_true(dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[0, 1, ::, 0]), "C")),
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

    dr.assert_true(dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[1, 5, ::, 5]), "C")),
                   "Incorrect values for radiance (a=1, t=6, lbda=5)")

@dr.syntax
def test_render(database: tuple[list[int], mi.Float], database_rad: tuple[list[int], mi.Float], render_shape, t, a, eta):
    # Compute coefficients
    params = get_params(database[1], database[0], t, a, eta)
    mean_radiance = get_params(database_rad[1], database_rad[0], t, a, eta)

    # Get rays
    view_dir, thetas = get_north_hemisphere_rays(render_shape, True)
    sun_dir = mi.Vector3f(dr.sin(eta), 0, dr.cos(eta))

    gammas = dr.safe_acos(dr.dot(view_dir, sun_dir))

    res = dr.zeros(mi.TensorXf, (render_shape[0], render_shape[1], 3))

    i = 0
    while i < mean_radiance.shape[0]:
        coefs = dr.gather(mi.Float, params, i*9 + dr.arange(mi.UInt32, 9))
        res[::, ::, i] = get_rad(coefs, thetas, gammas) * mean_radiance[i]
        i += 1

    return res

def render_sun():
    resolution = (1024, 1024*4)
    view_dir = get_north_hemisphere_rays(resolution)

    eta = 0.5 * dr.pi * 2/100

    sun_dir = mi.Vector3f(dr.cos(eta), 0, dr.sin(eta))

    res = dr.reshape(mi.TensorXf, get_sun(sun_dir, view_dir, dr.pi/5, 0.1, 1e-2), resolution)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/sun_test.exr", res)

def render_suite():
    database_rad = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb_rad.bin")
    database = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb.bin")

    shape = (256, 256*4)

    r70 = dr.pi/2 * (70/100)
    r50 = dr.pi/2 * (50/100)
    r30 = dr.pi/2 * (30/100)
    r10 = dr.pi/2 * (10/100)
    r5 = dr.pi/2 * (5/100)

    test_params = [(1, 0.5, r30), (3, 0.2, r30), (3, 0.8, r30), (5, 0, r30), (5, 1, r30), (6, 0.2, r30), (6, 0.5, r30)]

    for (t, a, eta) in test_params:
        res = test_render(database, database_rad, shape, t, a, eta)
        mi.util.write_bitmap(f"sunsky-testing/res/renders/sm_t{t}_a{a}_eta{int(eta * 2 * dr.inv_pi * 100)}.exr", res)

    for eta in [r5, r10, r30, r50, r70]:
        t, a = 6, 0.5
        res = test_render(database, database_rad, shape, t, a, eta)
        mi.util.write_bitmap(f"sunsky-testing/res/renders/sm_t{t}_a{a}_eta{int(eta * 2 * dr.inv_pi * 100)}.exr", res)

def test_plot_spectral():
    wavelengths = [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]
    dataset_rad: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec_rad.bin")
    dataset: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec.bin")

    eta, t, a = dr.pi/2 * (30/100), 6, 0.5

    res = test_render(dataset, dataset_rad, (256, 256*3), t, a, eta)

    fig, axes = plt.subplots(3, 4)

    for i in range(11):
        subplot = axes[i//4][i%4]
        subplot.imshow(res[i], cmap="hot")
        subplot.axis('off')
        subplot.set_title(f"{wavelengths[i]}$\\lambda$")
    fig.delaxes(axes[2][3])

    fig.suptitle("Sky model render in different wavelengths")
    plt.show()


if __name__ == "__main__":
    #mi.write_sky_model_data_v1("sunsky-testing/res/datasets/ssm_dataset")

    test_mean_radiance_data()
    test_radiance_data()
    render_suite()

    #database_rad = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb_rad.bin")
    #database = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb.bin")
#
    #render_shape = (256, 256*4)
#
    #dr.kernel_history_clear()
#
    #with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
    #    res = test_render(database, database_rad, render_shape, 2, 0.5, dr.pi/4)
#
    #dr.print(dr.kernel_history())



