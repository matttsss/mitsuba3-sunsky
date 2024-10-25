import mitsuba as mi
import drjit as dr

def get_sun(sun_dir, view_dir, sun_radii, sun_dropoff, horizon_dropoff):
    def smoothstep(x):
        x = dr.clip(x, 0, 1)
        return 3 * x**2 - 2 * x**3

    theta = dr.acos(mi.Frame3f(mi.Vector3f(0, 0, 1)).cos_theta(view_dir))
    eta = dr.acos(dr.dot(sun_dir, view_dir))

    return smoothstep((eta - 0.5 * sun_radii) / -sun_dropoff) * smoothstep((theta - 0.5 * dr.pi) / -horizon_dropoff)


def bezier_interpolate(dataset: mi.Float, block_size: int, offset: mi.UInt32, x: mi.Float) -> mi.Float:
    """
     Interpolates data along a quintic Bézier curve
    :param dataset: Flat dataset
    :param block_size: Size of the block (NB_COLORS [* NB_PARAMS])
    :param offset: Offset to the start of the block
    :param x: belongs to [0, 1], point to interpolate data at
    :return: The interpolated data in flat array of shape: (NB_COLORS [, NB_PARAMS])
    """

    coefs  = [1, 5, 10, 10, 5, 1]
    indices = offset + dr.arange(mi.UInt32, block_size)

    res = dr.zeros(mi.Float, block_size)
    for i in range(6):
        data = dr.gather(mi.Float, dataset, indices + i * block_size)
        res += coefs[i] * (x ** i) * ((1 - x) ** (5-i)) * data

    return res


def get_params(dataset: mi.Float, t: mi.Int | mi.Float, a: mi.Int | mi.Float, eta: mi.Float) -> mi.Float:
    """
    Get the coefficients for the sky model
    :param dataset: Main dataset to interpolate from
                    (shape: (ALBEDO, TURBIDITY, CTRL_PTS, NB_COLORS [, NB_PARAMS]))
    :param t: Turbidity setting (1-10)
    :param a: Albedo setting (0-1)
    :param eta: Sun elevation angle
    :return: Interpolated coefficients / mean radiance values
    """
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(dr.all((0 <= a) & (a <= 1)), "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    t = dr.clip(t, 1, 10)
    a = dr.clip(a, 0, 1)
    eta = dr.clip(eta, 0.0, 0.5 * dr.pi)

    x: mi.Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    t_int  = mi.UInt32(dr.floor(t))
    t_low  = dr.maximum(t_int - 1, 0)
    t_high = dr.minimum(t_low + 1, 9)
    t_rem  = t - t_int

    t_block_size = len(dataset) // 10
    a_block_size = t_block_size // 2
    ctrl_block_size = a_block_size // 6

    res_a_low = dr.lerp(bezier_interpolate(dataset, ctrl_block_size, t_low  * t_block_size + 0 * a_block_size, x),
                        bezier_interpolate(dataset, ctrl_block_size, t_high  * t_block_size + 0 * a_block_size, x), t_rem)
    res_a_high = dr.lerp(bezier_interpolate(dataset, ctrl_block_size, t_low  * t_block_size + 1 * a_block_size, x),
                         bezier_interpolate(dataset, ctrl_block_size, t_high  * t_block_size + 1 * a_block_size, x), t_rem)

    inner_block_size = (ctrl_block_size // (3 if dr.hint(mi.is_rgb, mode="scalar") else 11))

    res = dr.lerp(res_a_low, res_a_high, dr.repeat(a, inner_block_size))
    return res & ((1 <= t <= 10) & (0 <= eta <= 0.5 * dr.pi))
