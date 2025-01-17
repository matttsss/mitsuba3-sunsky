import itertools
import sys

sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

from render_sky_scene import render_scene

def eval_full_spec(plugin, si, wavelengths, render_res = (512, 1024)):
    nb_channels = len(wavelengths)

    output_image = dr.zeros(mi.Float, render_res[0] * render_res[1] * nb_channels)
    for i, lbda in enumerate(wavelengths):
        si.wavelengths = lbda
        res = plugin.eval(si)[0]
        dr.scatter(output_image, res, i + nb_channels * dr.arange(mi.UInt32, render_res[0] * render_res[1]))

    return mi.TensorXf(output_image, (*render_res, nb_channels))

def check_spectral_constants():

    def make_range_idx(nb, start, end):
        return [(i/(nb-1)) * (end - start) + start for i in range(nb)]

    etas = make_range_idx(5, 0, 0.5 * dr.pi)
    turbs = make_range_idx(7, 1, 10)
    albedos = make_range_idx(3, 0, 1)

    phi_sun = 0

    ratio_list = []
    for (eta, turb, albedo) in itertools.product(etas, turbs, albedos):

        # Render in spectral
        image = render_scene(turb, albedo, eta, phi_sun)
        lum = image[::, ::, 0] * 0.212671 + image[::, ::, 1] * 0.715160 + image[::, ::, 2] * 0.072169
        max_lum = dr.max(lum)

        # Load reference RGB image
        path = f"../renders/rgb/sky_rgb_eta{eta:.3f}_t{turb:.3f}_a{albedo:.3f}.exr"
        ref_image = mi.TensorXf(mi.Bitmap(path))
        ref_lum = ref_image[::, ::, 0] * 0.212671 + ref_image[::, ::, 1] * 0.715160 + ref_image[::, ::, 2] * 0.072169
        ref_max_lum = dr.max(ref_lum)

        ratio_list.append(dr.ravel(max_lum / ref_max_lum)[0])
        #dr.print("For turbidity {t}, albedo {a}, eta {eta}:", t=turb, a=albedo, eta=dr.rad2deg(eta))
        #dr.print("Ratio for normalization: {ratio}\n", ratio=normalization / ref_normalization)

        #fig, ax = plt.subplots(2)
        #ax[0].imshow(image)
        #ax[1].imshow(lum)
        #plt.show()

    ratios = mi.Float(ratio_list)
    dr.print("Mean ratio: {mean}", mean=dr.mean(ratios))
    dr.print("Max ratio: {max}", max=dr.max(ratios))
    dr.print("Min ratio: {min}", min=dr.min(ratios))

    std_dev = dr.sqrt(dr.sum((ratios - dr.mean(ratios))**2) / (len(ratios) - 1))
    dr.print("Standard deviation: {std}", std=std_dev)

def check_spectral_conversion():
    t, a = 3.2, 0.0
    eta = dr.deg2rad(48.2)
    phi_sun = -4*dr.pi/5

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    plugin = mi.load_dict({
        'type': 'sunsky',
        'sun_direction': [cp * st, sp * st, ct],
        'sun_scale': 0.0,
        'turbidity': t,
        'albedo': a,
    })

    si = dr.zeros(mi.SurfaceInteraction3f)
    if dr.hint(mi.is_spectral, mode="scalar"):
        si.wavelengths = dr.arange(mi.Float, 320, 760, 40)
    si.wi = -mi.Vector3f(cp * st, sp * st, ct)

    res = plugin.eval(si)

    if dr.hint(mi.is_spectral, mode="scalar"):
        res = dr.mean(mi.spectrum_to_srgb(res, si.wavelengths), axis=1)
        dr.print(res)

        lum = mi.luminance(res)
        dr.print(lum)
    else:
        lum = mi.luminance(res)
        dr.print(lum)

def check_comp_black_body():
    import matplotlib.pyplot as plt

    t, a = 5, 0.9
    eta = 0
    phi_sun = -4*dr.pi/5

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    sun_dir = mi.Vector3f(cp * st, sp * st, ct)
    plugin = mi.load_dict({
        'type': 'sunsky',
        'sun_direction': [cp * st, sp * st, ct],
        'sky_scale': 1.0,
        'turbidity': t,
        'albedo': a,
    })

    blackbody = mi.load_dict({
        'type': 'blackbody',
        'temperature': 5778,
    })

    sp, cp = dr.sincos(phi_sun - dr.deg2rad(0.1))
    st, ct = dr.sincos(dr.pi/2 - eta)

    start, end = 360, 830
    step = (end - start) / 20
    wavelengths = [start + step/2 + i*step for i in range(20)]
    #wavelengths = [320 + 40 * i for i in range(11)]
    #wavelengths = [405]
    si = dr.zeros(mi.SurfaceInteraction3f)
    if dr.hint(mi.is_spectral, mode="scalar"):
        si.wavelengths = mi.Float(wavelengths)
    si.wi = -mi.Vector3f(cp * st, sp * st, ct)

    sunsky_res = plugin.eval(si)
    blackbody_res = blackbody.eval(si)

    dr.print("Eta: {eta}, Gamma: {gamma}", eta=eta, gamma=dr.acos(dr.dot(sun_dir, -si.wi)))
    dr.print("Radiance: {rad}", rad=sunsky_res[0])

    ref_model = [0.0209954, 0.0602066, 0.377907, 0.870337, 2.32443, 4.21748, 6.78515, 10.7099, 15.2494, 25.6342, 36.2361, 47.1432, 62.8167, 81.9069, 106.086, 0, 0, 0, 0, 0]
    plt.plot(wavelengths, sunsky_res[0].numpy(), label="Sunsky")
    plt.plot(wavelengths, blackbody_res[0].numpy(), label="Blackbody (No atmosphere)")
    plt.plot(wavelengths, ref_model, label="Reference")

    ref_model = mi.Float(ref_model)
    dr.print("Relative error: {err}", err=dr.abs(sunsky_res[0] - ref_model) / (ref_model + 1e-5))

    dr.print("Model color: {model_color}, Ref color: {ref_color}",
             model_color=dr.mean(mi.spectrum_to_srgb(sunsky_res, si.wavelengths), axis=1),
             ref_color=dr.mean(mi.spectrum_to_srgb(ref_model, si.wavelengths[0]), axis=1))
    #matching_func = mi.linear_rgb_rec(si.wavelengths[0])
    #plt.plot(wavelengths, matching_func[0].numpy(), label="Red color matching function", color="red")
    #plt.plot(wavelengths, matching_func[1].numpy(), label="Green color matching function", color="green")
    #plt.plot(wavelengths, matching_func[2].numpy(), label="Blue color matching function", color="blue")

    plt.axvline(720, linestyle='--', color="red", label="Last valid wavelength")
    plt.xlabel("Wavelengths[nm]")
    plt.ylabel(r"Radiance[$W/(m^2 \cdot sr \cdot nm)$]")
    plt.legend()
    plt.show()


def check_spec_film():
    from rendering.spherical_sensor import SphericalSensor

    t, a = 1, 0.0
    eta = dr.deg2rad(0)
    phi_sun = -4*dr.pi/5

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    film = {
        'type': 'specfilm',
        'width': 1024,
        'height': 512,
    }

    start, end = 360, 830
    step = (end - start) / 10
    lbda = start - step/2
    for i in range(10):
        lbda += step
        film[f'band_{i:02d}'] = {
            'type': 'regular',
            'wavelength_min': lbda,
            'wavelength_max': lbda + 0.0001,
            'values': '1, 1'
        }
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'direct'
        },
        'sensor': {
            'type': 'spherical',
            'sampler': {
                'type': 'independent'
            },
            'film': film
        },
        'emitter': {
            'type': 'sunsky',
            'sun_direction': [cp * st, sp * st, ct],
            'sun_scale': 0.0,
            'sky_scale': 1.0,
            'turbidity': t,
            'albedo': a,
        }
    })

    image = mi.Bitmap(mi.render(scene, spp=8000), pixel_format=mi.Bitmap.PixelFormat.MultiChannel)
    mi.util.write_bitmap("sunsky-testing/res/renders/full_channels.exr", image)

def check_full_eval():
    from helpers import get_camera_rays

    t, a = 2.5, 0.0
    eta = dr.deg2rad(0)
    phi_sun = -4*dr.pi/5

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    plugin = mi.load_dict({
        'type': 'sunsky',
        'sun_direction': [cp * st, sp * st, ct],
        'sun_scale': 1.0,
        'sky_scale': 0.0,
        'turbidity': t,
        'albedo': a,
    })

    render_res = (512, 1024)

    start, end = 360, 830
    step = (end - start) / 10
    wavelengths = [start + step/2 + i*step for i in range(10)]

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -get_camera_rays(mi.Vector3f(cp * st, sp * st, ct), (0.1, 0.1), render_res)

    output_image = eval_full_spec(plugin, si, wavelengths, render_res)

    mi.util.write_bitmap("sunsky-testing/res/renders/full_channels.exr", output_image)

if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")
    #check_spectral_constants()
    check_comp_black_body()
    #check_spectral_conversion()

    #check_spec_film()
    #check_full_eval()
