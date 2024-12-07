import itertools
import sys
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

    for (eta, turb, albedo) in itertools.product(etas, turbs, albedos):

        image = render_scene(turb, albedo, eta, phi_sun)
        lum = image[::, ::, 0] * 0.212671 + image[::, ::, 1] * 0.715160 + image[::, ::, 2] * 0.072169
        normalization = 1 / dr.max(lum)
        print("For turbidity {t}, albedo {a}, eta {eta}:".format(t=turb, a=albedo, eta=dr.rad2deg(eta)))
        print("Normalization constant: {norm}\n".format(norm=normalization))

        fig, ax = plt.subplots(2)
        ax[0].imshow(image)
        ax[1].imshow(lum)
        plt.show()


if __name__ == "__main__":
    test_spectral_constants()

