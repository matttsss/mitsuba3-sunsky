import pytest

import sys
sys.path.insert(0, "build/python")

import drjit as dr
import numpy as np

from drjit.llvm import Float, Int

def dr_quintic_bezier(data, x):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [6, ...]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data
    """
    coefs = Float(1, 5, 10, 10, 5, 1)
    powers = Int(0, 1, 2, 3, 4, 5)
    powers_flipped = Int(5, 4, 3, 2, 1, 0)

    coefs *= dr.power(x, powers) * dr.power(1 - x, powers_flipped)
    return dr.sum(coefs * data)

@dr.syntax
def dr_quintic_bezier_bis(data, x):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [6, ...]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data
    """

    coefs = Float(1, 5, 10, 10, 5, 1)
    x_inv = 1 - x

    x_inv_ = x_ = 1

    i = Int(0)
    while i < 6:
        coefs[i] *= x_
        coefs[5-i] *= x_inv_
        x_ *= x
        x_inv_ *= x_inv

        i += 1

    return dr.sum(coefs * data)

def np_quintic_bezier(data, x):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [6, ...]
    :param x: belongs to [0, 1], Point to interpolate data at
    :return: The interpolated data
    """

    # Get the polynomial coefficients
    coefs = np.vander([x, (1-x)], 6)

    # Fuse the coefficients
    coefs = coefs[0, ::-1] * coefs[1] * np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])

    # Interpolate the data
    return np.sum(coefs * data, axis=0)


def bezier_interpolate(data, eta, t, a):
    # TODO sort data shape
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(0 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x = dr.power(2 * dr.inv_pi * eta, 1/3)

    # Lerp for albedo and turbidity
    if not dr.is_integral_v(t) and not dr.is_integral_v(a):
        top_start = dr_quintic_bezier(data[dr.floor(t), 1], x)
        top_end   = dr_quintic_bezier(data[dr.ceil(t),  1], x)
        bot_start = dr_quintic_bezier(data[dr.floor(t), 0], x)
        bot_end   = dr_quintic_bezier(data[dr.ceil(t),  0], x)

        lerp_a_top = dr.lerp(top_start, top_end, a)
        lerp_a_bot = dr.lerp(bot_start, bot_end, a)

        return dr.lerp(lerp_a_bot, lerp_a_top, t - dr.floor(t))

    # Lerp for albedo
    if dr.is_integral_v(t) and not dr.is_integral_v(a):
        d1 = dr_quintic_bezier(data[t, 0], x)
        d2 = dr_quintic_bezier(data[t, 1], x)

        return dr.lerp(d1, d2, a)

    # Lerp for turbidity
    if not dr.is_integral_v(t) and dr.is_integral_v(a):
        d1 = dr_quintic_bezier(data[dr.floor(t), a], x)
        d2 = dr_quintic_bezier(data[dr.ceil(t), a], x)

        return dr.lerp(d1, d2, t - dr.floor(t))

    # Evaluate on data point
    if dr.is_integral_v(t) and dr.is_integral_v(a):
        return dr_quintic_bezier(data[t, a], x)


#print(dr_quintic_bezier(None, 0.25))
#print(np_quintic_bezier(None, 0.25))
