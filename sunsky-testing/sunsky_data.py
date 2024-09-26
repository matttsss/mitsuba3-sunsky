import sys
sys.path.insert(0, "build/python")

import drjit as dr

from drjit.cuda import Float, Int, TensorXf

def dr_quintic_bezier(data: TensorXf, x: Float):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [N_wavelengths, 9, 6]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data of shape [N_wavelengths, 9]
    """

    coefs = Float(1, 5, 10, 10, 5, 1)
    powers = Int(0, 1, 2, 3, 4, 5)
    powers_flipped = Int(5, 4, 3, 2, 1, 0)

    coefs *= dr.power(x, powers) * dr.power(1 - x, powers_flipped)

    return dr.sum(data * coefs, axis = data.ndim - 1)


def bezier_interpolate(data: TensorXf, eta: Float, t: Int | Float, a: Int | Float):
    # TODO sort data shape
    dr.assert_true(data.shape[1] == 10 and data.shape[2] == 2, "Sky model dataset is not not of the right shape")
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x: Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    # Lerp for albedo and turbidity
    if not dr.is_integral_v(t) and not dr.is_integral_v(a):
        top_start = dr_quintic_bezier(data[::, dr.floor(t), 1], x)
        top_end   = dr_quintic_bezier(data[::, dr.ceil(t),  1], x)
        bot_start = dr_quintic_bezier(data[::, dr.floor(t), 0], x)
        bot_end   = dr_quintic_bezier(data[::, dr.ceil(t),  0], x)

        lerp_a_top = dr.lerp(top_start, top_end, a)
        lerp_a_bot = dr.lerp(bot_start, bot_end, a)

        return dr.lerp(lerp_a_bot, lerp_a_top, t - dr.floor(t))

    # Lerp for albedo
    if dr.is_integral_v(t) and not dr.is_integral_v(a):
        d1 = dr_quintic_bezier(data[::, t, 0], x)
        d2 = dr_quintic_bezier(data[::, t, 1], x)

        return dr.lerp(d1, d2, a)

    # Lerp for turbidity
    if not dr.is_integral_v(t) and dr.is_integral_v(a):
        d1 = dr_quintic_bezier(data[::, dr.floor(t), a], x)
        d2 = dr_quintic_bezier(data[::, dr.ceil(t), a], x)

        return dr.lerp(d1, d2, t - dr.floor(t))

    # Evaluate on data point
    if dr.is_integral_v(t) and dr.is_integral_v(a):
        return dr_quintic_bezier(data[::, t, a], x)


print(dr_quintic_bezier(dr.ones(TensorXf, (11, 9, 6)), Float(0.25)).shape)
print(bezier_interpolate(dr.ones(TensorXf, (11, 10, 2, 9, 6)), Float(0.25), Int(5), Int(0)).shape)
