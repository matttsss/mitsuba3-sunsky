import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant("llvm_rgb")

from helpers import get_north_hemisphere_rays
from test_sky_data import test_radiance_data, test_mean_radiance_data
from sunsky_data import get_params, get_rad, get_sun

@dr.syntax
def test_render(database, database_rad, render_shape, t, a, eta):
    # Compute coefficients
    params = get_params(database, t, a, eta)
    mean_radiance = get_params(database_rad, t, a, eta)

    # Get rays
    view_dir, thetas = get_north_hemisphere_rays(render_shape, True)
    sun_dir = mi.Vector3f(dr.sin(eta), 0, dr.cos(eta))

    gammas = dr.safe_acos(dr.dot(view_dir, sun_dir))

    nb_channels = mean_radiance.shape[0]
    res = dr.zeros(mi.TensorXf, (render_shape[1], render_shape[0], nb_channels))

    i = 0
    while i < nb_channels:
        coefs = dr.gather(mi.Float, params, i*9 + dr.arange(mi.UInt32, 9))
        res[::, ::, i] = get_rad(coefs, thetas, gammas) * mean_radiance[i]
        i += 1

    return res

def render_sun():
    resolution = (1024*4, 1024)
    view_dir = get_north_hemisphere_rays(resolution)

    eta = 0.5 * dr.pi * 2/100

    sun_dir = mi.Vector3f(dr.cos(eta), 0, dr.sin(eta))

    res = dr.reshape(mi.TensorXf, get_sun(sun_dir, view_dir, dr.pi/5, 0.1, 1e-2), resolution)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/sun_test.exr", res)


def render_suite():
    _, dataset_rad = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb_rad.bin")
    _, dataset = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_rgb.bin")

    shape = (256*4, 256)

    r70 = dr.pi/2 * (70/100)
    r50 = dr.pi/2 * (50/100)
    r30 = dr.pi/2 * (30/100)
    r10 = dr.pi/2 * (10/100)
    r5 = dr.pi/2 * (5/100)

    test_params = [(1, 0.5, r30), (3, 0.2, r30), (3, 0.8, r30), (5, 0, r30), (5, 1, r30), (6, 0.2, r30), (6, 0.5, r30)]

    for (t, a, eta) in test_params:
        res = test_render(dataset, dataset_rad, shape, t, a, eta)
        mi.util.write_bitmap(f"sunsky-testing/res/renders/sm_t{t}_a{a}_eta{int(eta * 2 * dr.inv_pi * 100)}.exr", res)

    for eta in [r5, r10, r30, r50, r70]:
        t, a = 6, 0.5
        res = test_render(dataset, dataset_rad, shape, t, a, eta)
        mi.util.write_bitmap(f"sunsky-testing/res/renders/sm_t{t}_a{a}_eta{int(eta * 2 * dr.inv_pi * 100)}.exr", res)

def test_plot_spectral():
    wavelengths = [320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]
    _, dataset_rad = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec_rad.bin")
    _, dataset = mi.array_from_file("sunsky-testing/res/datasets/ssm_dataset_v1_spec.bin")

    eta, t, a = dr.pi/2 * (30/100), 6, 0.5

    res = test_render(dataset, dataset_rad, (256*3, 256), t, a, eta)

    fig, axes = plt.subplots(3, 4)

    for i in range(11):
        subplot = axes[i//4][i%4]
        subplot.imshow(res[::, ::, i], cmap="hot")
        subplot.axis('off')
        subplot.set_title(f"{wavelengths[i]}$\\lambda$")
    fig.delaxes(axes[2][3])

    fig.suptitle("Sky model render in different wavelengths")
    plt.show()


if __name__ == "__main__":
    mi.write_sky_model_data_v1("sunsky-testing/res/datasets/ssm_dataset")

    test_mean_radiance_data()
    test_radiance_data()
    test_plot_spectral()
    render_suite()
