import sys
sys.path.insert(0, "build/python")

import matplotlib.pyplot as plt

import mitsuba as mi
import drjit as dr

mi.set_variant("llvm_ad_spectral")

def bezier_interpolate(data: mi.TensorXf, x: mi.Float):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [6, ...]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data
    """

    coefs = mi.Float(1, 5, 10, 10, 5, 1)
    powers = dr.arange(mi.Int, 6)

    coefs *= dr.power(x, powers)
    coefs *= dr.power(1 - x, dr.reverse(powers))

    return dr.sum(coefs * data, axis=data.ndim - 1)


def get_params(database: mi.TensorXf, eta: mi.Float, t: mi.Int | mi.Float, a: mi.Int | mi.Float):
    dr.assert_true(database.shape[0] == 2 and database.shape[1] == 10, "Sky model dataset is not not of the right shape")
    dr.assert_true(0 <= eta < 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x: mi.Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    # Lerp for albedo and turbidity
    if not dr.is_integral_v(t) and not dr.is_integral_v(a):
        t_idx_low = dr.floor(t)
        t_idx_low = t_idx_low if not t_idx_low else t_idx_low - 1

        top_start = bezier_interpolate(database[1, t_idx_low], x)
        top_end   = bezier_interpolate(database[1, t_idx_low + 1], x)
        bot_start = bezier_interpolate(database[0, t_idx_low], x)
        bot_end   = bezier_interpolate(database[0, t_idx_low + 1], x)

        lerp_a_top = dr.lerp(top_start, top_end, a)
        lerp_a_bot = dr.lerp(bot_start, bot_end, a)

        return dr.lerp(lerp_a_bot, lerp_a_top, t - dr.floor(t))

    # Lerp for albedo
    if dr.is_integral_v(t) and not dr.is_integral_v(a):
        d1 = bezier_interpolate(database[0, t-1], x)
        d2 = bezier_interpolate(database[1, t-1], x)

        return dr.lerp(d1, d2, a)

    # Lerp for turbidity
    if not dr.is_integral_v(t) and dr.is_integral_v(a):
        t_idx_low = dr.floor(t)
        t_idx_low = t_idx_low if not t_idx_low else t_idx_low - 1

        d1 = bezier_interpolate(database[a, t_idx_low], x)
        d2 = bezier_interpolate(database[a, t_idx_low + 1], x)

        return dr.lerp(d1, d2, t - dr.floor(t))

    # Evaluate on data point
    if dr.is_integral_v(t) and dr.is_integral_v(a):
        return bezier_interpolate(database[a, t], x)


def test_mean_radiance_data():
    dataset_rad: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v3_spec_rad.bin")

    # Control points for mean radiance at a = 0, t = 2, lbda = 0
    expected = mi.Float(9.160628e-004, 2.599956e-004, 5.466998e-003, -1.503537e-002, 7.200167e-002, 5.387713e-002)
    dr.allclose(expected, dataset_rad[0, 1, 0])

    # Control points for mean radiance at a = 1, t = 6, lbda = 5
    expected = mi.Float(1.245635e-002, 2.874175e-002, -6.384005e-002, 2.429023e-001, 2.428387e-001, 2.418906e-001,)
    dr.allclose(expected, dataset_rad[1, 5, 5])

def test_radiance_data():
    dataset: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v3_spec.bin")

    # Control points for mean radiance at a = 0, t = 2, lbda = 0
    expected = mi.Float(-1.298333e+001, -3.775577e+000, -5.173531e+000, 5.316518e+000, -2.572615e-002, 1.516601e-001,
                        -8.297168e-006, 1.669649e+000, 9.000495e-001, -1.402639e+001, -3.787558e+000, 7.611941e-002,
                        2.521881e-001, -5.859973e-002, 1.753711e-001, 4.670097e-005, 1.459275e+000, 8.998629e-001,
                        -2.190256e+000, -3.575495e+000, -4.930996e-001, 4.826321e-002, -6.797145e-002, 3.425922e-002,
                        -3.512550e-004, 1.978419e+000, 8.866517e-001, -2.415991e+000, -1.453294e+000, 2.170671e-001,
                        1.341284e-001, -1.926330e-001, 1.059103e-001, 1.360739e-003, 1.587725e+000, 9.821154e-001,
                        -5.254592e-001, -8.181026e-001, 7.535702e-001, -3.323364e-002, 4.503149e-001, 5.778285e-001,
                        -4.089673e-003, 3.335089e-001, 6.827164e-001, -1.280108e+000, -1.013716e+000, 5.577676e-001,
                        9.539205e-004, -4.934956e+000, 2.642883e-001, 1.005169e-002, 9.265844e-001, 4.999698e-001)

    dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[0, 1, ::, 0]), "C"))

    # Control points for mean radiance at a = 1, t = 6, lbda = 5
    expected = mi.Float(-1.330626e+000, -4.272516e-001, -1.317682e+000, 1.650847e+000, -1.192771e-001, 4.904191e-001,
                        4.074827e-002, 3.015846e+000, 5.271835e-001, -1.711989e+000, -8.644776e-001, 4.057135e-001,
                        9.037139e-001, -3.100084e-001, 6.317697e-002, 1.065625e-001, 9.226240e-002, 3.022474e-001,
                        -8.465894e-001, 1.652715e-001, -2.532361e-001, -2.422693e+000, 3.144841e-001, 1.839347e+000,
                        -2.818162e-001, 7.856667e+000, 1.387977e+000, -1.192114e+000, -3.830569e-001, 5.124751e-001,
                        7.280034e+000, -2.610477e+000, -1.832768e+000, 9.101904e-001, -3.349116e+000, -7.313079e-002,
                        -1.011026e+000, -1.061217e-001, 1.357854e+000, -1.496195e+001, -2.180975e+000, 2.484329e+000,
                        -3.239225e-001, 3.899425e+000, 1.179264e+000, -1.106228e+000, -1.927917e-001, 1.179701e+000,
                        2.379834e+001, -4.870211e+000, -1.290713e+000, 2.854422e-001, 2.078973e+000, 5.128625e-001)

    dr.allclose(expected, dr.unravel(mi.Float, mi.Float(dataset[1, 5, ::, 5]), "C"))


def get_rad(coefs: mi.TensorXf, theta: mi.Float, gamma: mi.Float):
    cos_theta = dr.cos(theta)
    cos_gamma = dr.cos(gamma)
    cos_gamma_sqr = dr.square(cos_gamma)

    A, B, C, D, E, F, G, H, I = coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], coefs[7], coefs[8]

    c1 = 1 + A * dr.exp(B / (cos_theta + 0.01))
    chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(H) - 2 * H * cos_gamma, 3/2)
    c2 = C + D * dr.exp(E * gamma) + F * cos_gamma_sqr + G * chi + I * dr.sqrt(cos_theta)

    return c1 * c2


@dr.syntax
def test_compute():
    dataset_rad: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v3_spec_rad.bin")
    dataset: mi.TensorXf = mi.tensor_from_file("sunsky-testing/res/ssm_dataset_v3_spec.bin")

    solar_elevation = dr.pi / 7
    params = get_params(dataset, solar_elevation, 2, 1)

    mean_radiance = get_params(dataset_rad, solar_elevation, 2, 1)

    gammas, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.pi, 256 * 3),
        dr.linspace(mi.Float, 0, dr.pi / 2 - 1e-5, 256)
    )
    thetas -= solar_elevation


    res = dr.zeros(mi.TensorXf, (256*3, 256, 11))

    idx = 0
    nb_samples = 256 * 256 * 3
    while idx < nb_samples:
        res[idx % 256*3, idx // 256*3] = get_rad(params, thetas[idx], gammas[idx])
        idx += 1

    plt.imshow(res[0], cmap='plasma')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    #mi.write_sky_model_data_v3("sunsky-testing/res/ssm_dataset")
    test_mean_radiance_data()
    test_radiance_data()
    test_compute()