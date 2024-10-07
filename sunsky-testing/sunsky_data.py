import mitsuba as mi
import drjit as dr

def get_sun(sun_dir, view_dir, sun_radii, sun_dropoff, horizon_dropoff):
    def smoothstep(x):
        x = dr.clip(x, 0, 1)
        return 3 * x**2 - 2 * x**3

    theta = dr.acos(mi.Frame3f(mi.Vector3f(0, 0, 1)).cos_theta(view_dir))
    eta = dr.acos(dr.dot(sun_dir, view_dir))

    return smoothstep((eta - 0.5 * sun_radii) / -sun_dropoff) * smoothstep((theta - 0.5 * dr.pi) / -horizon_dropoff)

def get_rad(coefs: mi.Float, theta: mi.Float, gamma: mi.Float) -> mi.Float:
    cos_theta = dr.cos(theta)
    cos_gamma = dr.cos(gamma)
    cos_gamma_sqr = dr.square(cos_gamma)

    c1 = 1 + coefs[0] * dr.exp(coefs[1] / (cos_theta + 0.01))
    chi = (1 + cos_gamma_sqr) / dr.power(1 + dr.square(coefs[7]) - 2 * coefs[7] * cos_gamma, 1.5)
    c2 = coefs[2] + coefs[3] * dr.exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[8] * dr.sqrt(cos_theta)

    return c1 * c2


def bezier_interpolate(dataset: mi.Float, block_size: int, offset: mi.UInt32, x: mi.Float) -> mi.Float:
    """
     Interpolates data along a quintic Bézier curve
    :param dataset: Flat dataset
    :param block_size: Size of the block (nb_colors * nb_parameters * nb_control_pts)
    :param offset: Offset to the start of the block
    :param x: belongs to [0, 1], point to interpolate data at
    :return: The interpolated data in flat array of "shape" [C, P]
    """

    coefs  = [1, 5, 10, 10, 5, 1]

    inner_block_size = block_size // 6
    indices = offset + dr.arange(mi.UInt32, inner_block_size)

    res = dr.zeros(mi.Float, inner_block_size)
    for i in range(6):
        data = dr.gather(mi.Float, dataset, indices + i * inner_block_size)
        res += coefs[i] * (x ** i) * ((1 - x) ** (5-i)) * data

    return res


def get_params(database: mi.Float, t: mi.Int | mi.Float, a: mi.Int | mi.Float, eta: mi.Float) -> mi.TensorXf:
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x: mi.Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    t_low = mi.UInt32(dr.floor(t - 1))
    t_high = dr.minimum(t_low + 1, 9)
    t_rem = t - t_low

    a_block_size = len(database) // 2
    t_block_size = a_block_size // 10

    res  = (1 - t_rem) * (1 - a) * bezier_interpolate(database, t_block_size, 0 * a_block_size + t_low  * t_block_size, x)
    res += (1 - t_rem) * a       * bezier_interpolate(database, t_block_size, 1 * a_block_size + t_low  * t_block_size, x)
    res += t_rem       * (1 - a) * bezier_interpolate(database, t_block_size, 0 * a_block_size + t_high * t_block_size, x)
    res += t_rem       * a       * bezier_interpolate(database, t_block_size, 1 * a_block_size + t_high * t_block_size, x)

    return res
