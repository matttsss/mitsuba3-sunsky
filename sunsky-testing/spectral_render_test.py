import itertools
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_spectral")

from helpers import get_north_hemisphere_rays
from render_scene import render_scene


def test_spectral_constants():

    def make_range_idx(nb, start, end):
        return [(i/(nb-1)) * (end - start) + start for i in range(nb)]

    etas = make_range_idx(5, 0, 0.5 * dr.pi)
    turbs = make_range_idx(7, 1, 10)
    albedos = make_range_idx(3, 0, 1)

    phi_sun = -4*dr.pi/5
    sp, cp = dr.sincos(phi_sun)

    wavelengths = mi.Float(320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720)

    rays = get_north_hemisphere_rays((1024, 256))
    wi_idx, wavelengths_idx = dr.meshgrid(dr.arange(mi.UInt32, 1024*256), dr.arange(mi.UInt32, 11))

    si = mi.SurfaceInteraction3f()
    si.wi = -rays
    #si.wavelengths = dr.gather(mi.Float, wavelengths, wavelengths_idx)

    for (eta, turb, albedo) in itertools.product(etas, turbs, albedos):
        #st, ct = dr.sincos(dr.pi/2 - eta)
        #sky = mi.load_dict({
        #    'type': 'sunsky',
        #    'sun_direction': [cp * st, sp * st, ct],
        #    'turbidity': turb,
        #    'albedo': albedo,
        #})

        image = render_scene(turb, albedo, eta, phi_sun).numpy()

        lum = image[::, ::, 0] * 0.212671 + image[::, ::, 1] * 0.715160 + image[::, ::, 2] * 0.072169
        fig, ax = plt.subplots(2)
        ax[0].imshow(image / np.max(lum))
        ax[1].imshow(lum)
        plt.show()
        print(lum.shape)
        print("For turbidity {t}, albedo {a}, eta {eta}:".format(t=turb, a=albedo, eta=dr.rad2deg(eta)))
        print("Normalization constant: {norm}".format(norm=np.max(lum)))
        print("\n\n")


if __name__ == "__main__":
    test_spectral_constants()

