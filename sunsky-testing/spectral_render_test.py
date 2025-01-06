import itertools
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

from render_sky_scene import render_scene


def test_spectral_constants():

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
        path = f"../renders/sky_rgb_eta{eta:.3f}_t{turb:.3f}_a{albedo:.3f}.exr"
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

def test_spectral_conversion():
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

def test_comp_black_body():
    t, a = 1, 0.0
    eta = dr.deg2rad(89.5)
    phi_sun = -4*dr.pi/5

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    plugin = mi.load_dict({
        'type': 'sunsky',
        'sun_direction': [cp * st, sp * st, ct],
        'turbidity': t,
        'albedo': a,
    })

    blackbody = mi.load_dict({
        'type': 'blackbody',
        'temperature': 5778,
    })

    si = dr.zeros(mi.SurfaceInteraction3f)
    if dr.hint(mi.is_spectral, mode="scalar"):
        si.wavelengths = dr.arange(mi.Float, 320, 760, 40)
    si.wi = -mi.Vector3f(cp * st, sp * st, ct)

    sunsky_res = plugin.eval(si)
    blackbody_res = blackbody.eval(si)

    dr.print("Sunsky luminance: {lum}", lum=dr.mean(mi.luminance(sunsky_res, si.wavelengths)))
    dr.print("Blackbody luminance: {lum}", lum=dr.mean(mi.luminance(blackbody_res, si.wavelengths)))


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    #test_spectral_constants()
    #test_comp_black_body()
    test_spectral_conversion()
