import pytest

import sys
sys.path.insert(0, "build/python")

import mitsuba as mi
mi.set_variant("llvm_ad_spectral")

from sunsky_plugin import *


spectrum_dicts = {
    'd65': {
        "type": "d65",
    },
    'regular': {
        "type": "regular",
        "wavelength_min": 500,
        "wavelength_max": 600,
        "values": "1, 2"
    }
}


def create_emitter_and_spectrum(s_key='d65'):
    emitter = mi.load_dict({
        "type" : "sunsky_emitter",
        "radiance" : spectrum_dicts[s_key]
    })
    spectrum = mi.load_dict(spectrum_dicts[s_key])
    expanded = spectrum.expand()
    if len(expanded) == 1:
        spectrum = expanded[0]

    return emitter, spectrum


@pytest.mark.parametrize("spectrum_key", spectrum_dicts.keys())
def chi2_test(variants_vec_spectral, spectrum_key):
    sse_dict = {
        'type' : 'sunsky_emitter',
        'radiance' : spectrum_dicts[spectrum_key]
    }

    sample_func, pdf_func = mi.chi2.EmitterAdapter("sunsky_emitter", sse_dict)
    chi2 = mi.chi2.ChiSquareTest(
        domain = mi.chi2.SphericalDomain(),
        sample_func = sample_func,
        pdf_func = pdf_func
    )

    assert chi2.run()

if __name__ == "__main__":
    chi2_test(None, "d65")
