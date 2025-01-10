import sys; sys.path.insert(0, "build/python")
import itertools

import drjit as dr
import mitsuba as mi

phi = dr.pi/5
SUN_HALF_APP = dr.deg2rad(0.25)

def get_plugin(turb, eta_ray, gamma, albedo = 0.0):
    theta_sun = dr.pi/2 - eta_ray - gamma
    theta_sun = dr.select(theta_sun < 0, dr.pi/2 - eta_ray + gamma, theta_sun)

    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(theta_sun)

    return mi.load_dict({
        'type': 'sunsky',
        'sun_direction':[cp * st, sp * st, ct],
        'sky_scale': 0.0,
        'turbidity': turb,
        'albedo': albedo,
    })

def compute_wav_sun(wavelengths, turb, eta_ray, gamma):
    ref_rad = mi.Float([mi.hosek_sun_rad(turb, wav, eta_ray, gamma) for wav in wavelengths])

    plugin = get_plugin(turb, eta_ray, gamma)

    si = mi.SurfaceInteraction3f()
    si.wavelengths = mi.Float(wavelengths)

    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(dr.pi/2 - eta_ray)
    si.wi = -mi.Vector3f(cp * st, sp * st, ct)

    res = plugin.eval(si)[0]
    relative_err = dr.abs(res - ref_rad) / (ref_rad + 1e-6)

    #dr.print("Radiance: {rad}", rad=res)
    #dr.print("Reference: {ref}", ref=ref_rad)
    #dr.print("Relative error, min: {min_err}, max: {max_err}, mean: {mean_err}",
    #         min_err=dr.min(relative_err),
    #         max_err=dr.max(relative_err),
    #         mean_err=dr.mean(relative_err)
    #         )
    return dr.mean(relative_err)

if __name__ == "__main__":
    mi.set_variant('cuda_ad_spectral')

    def make_range_idx(nb, a, b):
        return [(i/(nb-1)) * (b - a) + a for i in range(nb)]

    eps = 1e-3
    etas = make_range_idx(5, eps, dr.pi/2-eps)
    turbs = make_range_idx(7, 1, 10)
    gammas = make_range_idx(4, 0, SUN_HALF_APP-eps)

    start, end = 320, 720
    step = (end - start) / 20
    wavs = [start + step/2 + i*step for i in range(20)]

    i = 0
    max_err = 0
    max_err_turb, max_err_eta, max_err_gamma = -1, -1, -1

    for (eta, turb, gamma) in itertools.product(etas, turbs, gammas):
        #if i == 0:
        #    break
        err = compute_wav_sun(wavs, turb, eta, gamma)
        dr.print("Testing eta={eta}, turb={turb}, gamma={gamma}\n\t Error: {err}",
                 eta=eta, turb=turb, gamma=gamma, err=err)

        found_new_max = err > max_err
        max_err_turb = dr.select(found_new_max, turb, max_err_turb)
        max_err_eta = dr.select(found_new_max, eta, max_err_eta)
        max_err_gamma = dr.select(found_new_max, gamma, max_err_gamma)
        max_err = dr.select(err > max_err, err, max_err)
        i += 1

    dr.print("Max error: {err}, for turbidity {turb}, eta {eta}, gamma {gamma}", err=max_err, turb=max_err_turb, eta=max_err_eta, gamma=max_err_gamma)
    #compute_wav_sun(wavs, 10, 1.5708, 0.00436332)