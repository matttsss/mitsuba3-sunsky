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
    c2 = coefs[2] + coefs[3] * dr.exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[8] * dr.safe_sqrt(cos_theta)

    return c1 * c2


def bezier_interpolate(dataset: mi.Float, block_size: mi.UInt32, offset: mi.UInt32, x: mi.Float) -> mi.Float:
    """
     Interpolates data along a quintic Bézier curve
    :param dataset: Flat dataset
    :param block_size: Size of the block (nb_colors * nb_parameters * nb_control_pts)
    :param offset: Offset to the start of the block
    :param x: belongs to [0, 1], point to interpolate data at
    :return: The interpolated data in flat array of "shape" [C, P]
    """

    coefs  = [1, 5, 10, 10, 5, 1]
    powers = [0, 1, 2, 3, 4, 5]

    # TODO a way to avoid this? arange does not support a mi.UInt32 variant as size
    inner_block_size = (block_size//6)[0]
    indices = offset + dr.arange(mi.UInt32, inner_block_size)

    res = dr.zeros(mi.Float, inner_block_size)
    for i in range(6):
        data = dr.gather(mi.Float, dataset, indices + i * inner_block_size)
        res += coefs[i] * x**powers[i] * (1 - x)**powers[5-i] * data

    return res


def get_params(database: mi.Float, shape, t: mi.Int | mi.Float, a: mi.Int | mi.Float, eta: mi.Float) -> mi.TensorXf:
    dr.assert_true(shape[0] == 2 and shape[1] == 10, "Sky model dataset is not not of the right shape")
    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(0 <= a <= 1, "Albedo (a) value is not between 0 and 1: %f", a)
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 0 and 10: %f", t)

    x: mi.Float = dr.power(2 * dr.inv_pi * eta, 1/3)

    t_low = mi.UInt32(dr.floor(t - 1))

    a_offset_coef = mi.UInt32(dr.prod(shape)) // 2
    block_size = a_offset_coef // 10 # Size of the data block between two 't' indices

    # TODO condense lerps in while loop

    # Lerp on t_low
    t_offset = t_low * block_size
    start = bezier_interpolate(database, block_size, 0 * a_offset_coef + t_offset, x)
    end   = bezier_interpolate(database, block_size, 1 * a_offset_coef + t_offset, x)

    lerp_t_low = dr.lerp(start, end, a)

    if t_low == 9:
        return lerp_t_low

    # Lerp on t_high
    t_offset = (t_low + 1) * block_size
    start = bezier_interpolate(database, block_size, 0 * a_offset_coef + t_offset, x)
    end   = bezier_interpolate(database, block_size, 1 * a_offset_coef + t_offset, x)

    lerp_t_high = dr.lerp(start, end, a)

    return dr.lerp(lerp_t_low, lerp_t_high, t - dr.floor(t))
