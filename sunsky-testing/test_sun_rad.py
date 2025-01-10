import sys; sys.path.insert(0, "build/python")
import itertools

import drjit as dr
import mitsuba as mi

phi = dr.pi/5
SUN_HALF_APP = dr.deg2rad(0.5358/2.0)

def get_blackbody_rad(wavelengths):
    blackbody = mi.load_dict({
        'type': 'blackbody',
        'temperature': 5778,
    })
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wavelengths = mi.Float(wavelengths)

    return blackbody.eval(si)[0]



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
    return dr.mean(relative_err), res

def test_sun_rad_quadrature():

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
        err, _ = compute_wav_sun(wavs, turb, eta, gamma)
        dr.print("Testing eta={eta}, turb={turb}, gamma={gamma}\n\t Error: {err}",
                 eta=eta, turb=turb, gamma=gamma, err=err)

        found_new_max = err > max_err
        max_err_turb = dr.select(found_new_max, turb, max_err_turb)
        max_err_eta = dr.select(found_new_max, eta, max_err_eta)
        max_err_gamma = dr.select(found_new_max, gamma, max_err_gamma)
        max_err = dr.select(err > max_err, err, max_err)
        i += 1

    dr.print("Max error: {err}, for turbidity {turb}, eta {eta}, gamma {gamma}", err=max_err, turb=max_err_turb, eta=max_err_eta, gamma=max_err_gamma)
    #compute_wav_sun(wavs, 6.75, 9*dr.pi/10, 0.0)

def plot_some_radiance():
    import matplotlib.pyplot as plt

    start, end = 320, 720
    step = (end - start) / 20
    wavs = [start + step/2 + i*step for i in range(20)]

    params = [(3.2, dr.deg2rad(30), SUN_HALF_APP/6), (2, dr.deg2rad(22), SUN_HALF_APP/5), (5, dr.deg2rad(60), SUN_HALF_APP/10), (3, dr.deg2rad(1), 0.0)]
    param_str = [(r"30^{\circ}", r"\frac{\alpha}{12}"),
                 (r"22^{\circ}", r"\frac{\alpha}{10}"),
                 (r"60^{\circ}", r"\frac{\alpha}{20}"),
                 (r"1^{\circ}", r"0")]

    res = [compute_wav_sun(wavs, *param) for param in params]

    for (err, rad), (t, _, _), (eta_str, gamma_str) in zip(res, params, param_str):
        err = err[0]
        plt.plot(wavs, rad.numpy(), label=f'$t={t}$, $\eta={eta_str}$, $\gamma={gamma_str}$ MRE={err:.2e}')


    plt.plot(wavs, get_blackbody_rad(wavs).numpy(), label='Blackbody (5778K)')

    plt.legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(r'Radiance [$W/(m^2 \cdot sr \cdot nm)$]')

    plt.show()

if __name__ == "__main__":
    mi.set_variant('cuda_ad_spectral')

    #test_sun_rad_quadrature()
    plot_some_radiance()