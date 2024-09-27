import sys
sys.path.insert(0, "build/python")

import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_spectral")


def bezier_interpolate(data: mi.TensorXf, x: mi.Float):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [N_wavelengths, 9, 6]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data of shape [N_wavelengths, 9]
    """

    coefs = mi.Float(1, 5, 10, 10, 5, 1)
    powers = mi.Int(0, 1, 2, 3, 4, 5)
    powers_flipped = mi.Int(5, 4, 3, 2, 1, 0)

    coefs *= dr.power(x, powers) * dr.power(1 - x, powers_flipped)

    return dr.sum(data * coefs, axis = data.ndim - 1)


def get_params(database: mi.TensorXf, eta: mi.Float, t: mi.Int | mi.Float, a: mi.Int | mi.Float):
    dr.assert_true(database.shape[0] == 2 and database.shape[1] == 10, "Sky model dataset is not not of the right shape")
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x: mi.Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    # Lerp for albedo and turbidity
    if not dr.is_integral_v(t) and not dr.is_integral_v(a):
        t_idx_low = dr.floor(t)
        t_idx_low = t_idx_low if not t_idx_low else t_idx_low - 1
        t_idx_hig = t_idx_low + 1

        top_start = bezier_interpolate(database[1, t_idx_low], x)
        top_end   = bezier_interpolate(database[1, t_idx_hig], x)
        bot_start = bezier_interpolate(database[0, t_idx_low], x)
        bot_end   = bezier_interpolate(database[0, t_idx_hig], x)

        lerp_a_top = dr.lerp(top_start, top_end, a)
        lerp_a_bot = dr.lerp(bot_start, bot_end, a)

        return dr.lerp(lerp_a_bot, lerp_a_top, t - dr.floor(t))

    # Lerp for albedo
    if dr.is_integral_v(t) and not dr.is_integral_v(a):
        d1 = bezier_interpolate(database[0, t - 1], x)
        d2 = bezier_interpolate(database[1, t - 1], x)

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
        return bezier_interpolate(database[a, t - 1], x)


def compute_radiance(coefs: mi.TensorXf, theta: mi.Float, gamma: mi.Float):
    ...


def test():
    #mi.write_sky_model_data_v1("sunsky-testing/res/sunsky_dataset_spectral_v1")
    dataset_rad: mi.TensorXf = mi.read_sky_model_data("sunsky-testing/res/sunsky_dataset_spectral_v1.rad.bin")
    dataset: mi.TensorXf = mi.read_sky_model_data("sunsky-testing/res/sunsky_dataset_spectral_v1.bin")

    solar_elevation = dr.pi / 2

    mean_radiance = get_params(dataset_rad, solar_elevation, mi.Int(5), mi.Float(0.5))
    params = get_params(dataset, solar_elevation, mi.Int(5), mi.Int(0.5))




test()