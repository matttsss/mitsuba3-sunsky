import mitsuba as mi
import drjit as dr

def bezier_interpolate(data: mi.TensorXf, x: mi.Float):
    """
     Interpolates data along a quintic Bézier curve
    :param data: A tensor of shape [..., 6]
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

    # TODO sort for invisible dim when mi.Var is index

    # Lerp for albedo and turbidity
    if not dr.is_integral_v(t) and not dr.is_integral_v(a):
        t_idx_low = mi.Int(dr.floor(t) - 1)

        top_start = bezier_interpolate(database[1, t_idx_low]    , x)
        top_end   = bezier_interpolate(database[1, t_idx_low + 1], x)
        bot_start = bezier_interpolate(database[0, t_idx_low]    , x)
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
        t_idx_low = mi.Int(dr.floor(t) - 1)

        d1 = bezier_interpolate(database[a, t_idx_low], x)
        d2 = bezier_interpolate(database[a, t_idx_low + 1], x)

        return dr.lerp(d1, d2, t - dr.floor(t))

    # Evaluate on data point
    if dr.is_integral_v(t) and dr.is_integral_v(a):
        return bezier_interpolate(database[a, t], x)

def get_rad(coefs: mi.TensorXf, theta: mi.Float, gamma: mi.Float):
    cos_theta = dr.cos(theta)
    cos_gamma = dr.cos(gamma)
    cos_gamma_sqr = dr.square(cos_gamma)

    A, B, C, D, E, F, G, H, I = coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], coefs[7], coefs[8]

    c1 = 1 + A * dr.exp(B / (cos_theta + 0.01))
    chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(H) - 2 * H * cos_gamma, 1.5)
    c2 = C + D * dr.exp(E * gamma) + F * cos_gamma_sqr + G * chi + I * dr.safe_sqrt(cos_theta)

    return c1 * c2
