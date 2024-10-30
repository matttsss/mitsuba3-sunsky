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


def get_tgmm_table(dataset: mi.Float, t: mi.Float, eta: mi.Float):
    """
    Get the Truncated Gaussian Mixture Model table for the given parameters
    :param dataset: Main dataset to interpolate from (shape: (TURBIDITY, ETAs, NB_GAUSSIANS, NB_PARAMS))
    :param t: Turbidity of the model in [1, 10]
    :param eta: Elevation of the sun in [0, pi/2]
    :return: Interpolated TGMM table of shape (NB_GAUSSIANS, NB_PARAMS)
    """

    dr.assert_true(0 <= eta <= 0.5 * dr.pi, "Sun elevation is not between 0 and %f (pi/2): %f", (dr.pi/2, eta))
    dr.assert_true(1 <= t <= 10, "Turbidity value is not between 1 and 10: %f", t)

    eta = dr.rad2deg(eta)
    eta_idx_f = dr.maximum((eta - 2) / 3, 0) # adapt to table's discretization
    t_idx_f = dr.maximum(t - 2, 0)

    eta_idx_low = mi.UInt32(dr.floor(eta_idx_f))
    t_idx_low = mi.UInt32(dr.floor(t_idx_f))

    eta_idx_high = mi.UInt32(dr.minimum(eta_idx_low + 1, 29))
    t_idx_high = mi.UInt32(dr.minimum(t_idx_low + 1, 8))

    eta_rem = eta_idx_f - eta_idx_low
    t_rem = t_idx_f - t_idx_low

    t_block_size = len(dataset) // 9
    eta_block_size = t_block_size // 30

    idx_range = dr.arange(mi.UInt32, eta_block_size)

    res_t_low = dr.lerp(dr.gather(mi.Float, dataset, t_idx_low * t_block_size + eta_idx_low * eta_block_size + idx_range),
                        dr.gather(mi.Float, dataset, t_idx_low * t_block_size + eta_idx_high * eta_block_size + idx_range), eta_rem)
    res_t_high = dr.lerp(dr.gather(mi.Float, dataset, t_idx_high * t_block_size + eta_idx_low * eta_block_size + idx_range),
                         dr.gather(mi.Float, dataset, t_idx_high * t_block_size + eta_idx_high * eta_block_size + idx_range), eta_rem)

    return dr.lerp(res_t_low, res_t_high, t_rem)

#def gaussian_pdf(phi, theta, mu_p, mu_t, sigma_p, sigma_t, weight):
#    return weight * (1.0 / (2.0 * dr.pi * sigma_p * sigma_t)) * dr.exp( -0.5 * (((phi-mu_p)/sigma_p)**2 + ((theta-mu_t)/sigma_t)**2))

#def gaussian_truncated(phi, theta, mu_p, mu_t, sigma_p, sigma_t, weight):
#   return (gaussian_pdf(phi, theta, mu_p, mu_t, sigma_p, sigma_t, weight) /
#           ((gaussian_cdf(dr.two_pi, mu_p, sigma_p) - gaussian_cdf(0, mu_p, sigma_p)) * (gaussian_cdf(0.5 * dr.pi, mu_t, sigma_t) - gaussian_cdf(0, mu_t, sigma_t))))

def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + dr.erf(dr.inv_sqrt_two * (x - mu) / sigma))

def truncated_gaussian_quantile(sample, mu, sigma, a, b):
    temp = dr.lerp(gaussian_cdf(a, mu, sigma), gaussian_cdf(b, mu, sigma), sample)
    return dr.sqrt_two * sigma * dr.erfinv(2 * temp - 1) + mu

def tgmm_pdf(tgmm_table: mi.Float, theta: mi.Float, phi: mi.Float, active=True) -> mi.Float:
    """
    Evaluate the TGMM PDF at the given angles
    :param tgmm_table: TGMM table to evaluate
    :param theta: View zenith angle
    :param phi: Azimuthal angle, assuming sun is at phi = pi/2
    :return: PDF value
    """

    x = mi.Point2f(phi, theta)
    a = mi.Point2f(0.0)
    b = mi.Point2f(dr.two_pi, dr.pi/2)

    pdf = mi.Float(0.0)
    for i in range(5):
        coefs = dr.gather(mi.ArrayXf, tgmm_table, i, active, shape=(5, 1))

        mu = mi.Point2f(coefs[0], coefs[1])
        sigma = mi.Point2f(coefs[2], coefs[3])

        cdf_a = gaussian_cdf(a, mu, sigma)
        cdf_b = gaussian_cdf(b, mu, sigma)

        volume = (cdf_b[0] - cdf_a[0]) * (cdf_b[1] - cdf_a[1]) #    * (sigma[0] * sigma[1])

        unbounded_pdf = mi.warp.square_to_std_normal_pdf((x - mu) / sigma)

        pdf += coefs[4] * unbounded_pdf / volume

    return pdf & (theta >= 0) & (theta <= dr.pi / 2) & (phi >= 0) & (phi <= dr.two_pi)


def sample_tgmm(tgmm_table: mi.Float, sample: mi.Point2f, active=True) -> mi.Vector3f:
    """
    Sample the TGMM model at the given point
    :param tgmm_table: TGMM table containing the gaussian parameters
    :param sample: 2D sample point
    :return: Sampled point
    """

    dist = mi.DiscreteDistribution(dr.gather(mi.Float, tgmm_table, 5 * dr.arange(mi.UInt32, 5) + 4, active))
    dist.update()

    gaussian_idx, sample[0] = dist.sample_reuse(sample[0], active)

    gaussian = dr.gather(mi.ArrayXf, tgmm_table, gaussian_idx, active, shape=(5, 1))

    # (mu_p, mu_t)
    mu = mi.Point2f(gaussian[0], gaussian[1])

    # (sigma_p, sigma_t)
    sigma = mi.Point2f(gaussian[2], gaussian[3])

    a = mi.Point2f(0.0)
    b = mi.Point2f(dr.two_pi, dr.pi/2)

    cdf_a = gaussian_cdf(a, mu, sigma)
    cdf_b = gaussian_cdf(b, mu, sigma)

    sample = cdf_a + sample * (cdf_b - cdf_a)
    res = dr.sqrt_two * dr.erfinv(2 * sample - 1) * sigma + mu

    sp, cp = dr.sincos(res[0])
    st, ct = dr.sincos(dr.pi/2 - res[1])

    return mi.Vector3f(cp * st, sp * st, ct)