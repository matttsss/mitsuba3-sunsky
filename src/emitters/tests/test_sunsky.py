import pytest
import drjit as dr
import mitsuba as mi

def make_emitter(turb, sun_phi, sun_theta, albedo, sky_scale, sun_scale):
    sp_sun, cp_sun = dr.sincos(sun_phi)
    st, ct = dr.sincos(sun_theta)

    return mi.load_dict({
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sun_scale": sun_scale,
        "sky_scale": sky_scale,
        "turbidity": turb,
        "albedo_test": albedo
    })


def eval_full_spec(plugin, si, wavelengths, render_res = (512, 1024)):
    """
    Evaluates the plugin for the given surface interaction over all the given wavelengths
    :param plugin: Sunsky plugin to evaluate
    :param si: Surface interaction to evaluate the plugin with
    :param wavelengths: List of wavelengths to evaluate
    :param render_res: Resolution of the output image (the number of rays in SI should be equal to render_res[0] * render_res[1])
    :return: The output TensorXf image in format (render_res[0], render_res[1], len(wavelengths))
    """
    nb_channels = len(wavelengths)

    output_image = dr.zeros(mi.Float, render_res[0] * render_res[1] * nb_channels)
    for i, lbda in enumerate(wavelengths):
        si.wavelengths = lbda
        res = plugin.eval(si)[0]
        dr.scatter(output_image, res, i + nb_channels * dr.arange(mi.UInt32, render_res[0] * render_res[1]))

    return mi.TensorXf(output_image, (*render_res, nb_channels))


def make_range_idx(nb, a, b):
    return [(i/(nb-1)) * (b - a) + a for i in range(nb)]

@pytest.mark.parametrize("turb", make_range_idx(7, 1, 10))
@pytest.mark.parametrize("sun_eta", make_range_idx(5, 0, dr.pi/2))
@pytest.mark.parametrize("albedo", make_range_idx(3, 0, 1))
def test01_sky_radiance(variants_vec_backends_once, turb, sun_eta, albedo):
    render_res = (1024//2, 1024)

    plugin = make_emitter(turb=turb,
                          sun_phi=0,
                          sun_theta=dr.pi/2 - sun_eta,
                          albedo=albedo,
                          sun_scale=0.0,
                          sky_scale=1.0)

    # Generate the wavelengths
    start, end = 360, 830
    step = (end - start) / 10
    wavelengths = [start + step/2 + i*step for i in range(10)]

    # Generate the rays
    phis, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0.0, dr.two_pi, render_res[1]),
        dr.linspace(mi.Float, dr.pi, 0.0, render_res[0]))
    sp, cp = dr.sincos(phis)
    st, ct = dr.sincos(thetas)

    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(cp*st, sp*st, ct)

    # Evaluate the plugin
    rendered_scene  = mi.TensorXf(dr.ravel(plugin.eval(si)), (*render_res, 3)) if mi.is_rgb \
                 else eval_full_spec(plugin, si, wavelengths, render_res)

    render_type = "rgb" if mi.is_rgb else "spec"
    ref_path = f"../renders/{render_type}/sky_{render_type}_eta{sun_eta:.3f}_t{turb:.3f}_a{albedo:.3f}.exr"
    reference_scene = mi.TensorXf(mi.Bitmap(ref_path))

    rtol = 0.005 if mi.is_rgb else 0.0354
    rel_err = dr.mean(dr.abs(rendered_scene - reference_scene) / (dr.abs(reference_scene) + 0.001))

    assert rel_err <= rtol, f"Fail when rendering plugin: {plugin}\nMean relative error: {rel_err}, threshold: {rtol}"


@pytest.mark.parametrize("turb", make_range_idx(7, 1, 10))
@pytest.mark.parametrize("eta_ray", make_range_idx(5, 1e-3, dr.pi/2 - 1e-3))
@pytest.mark.parametrize("gamma", make_range_idx(4, 0, dr.deg2rad(0.5388/2.0) - 1e-3))
def test02_sun_radiance(variants_vec_backends_once, turb, eta_ray, gamma):
    wavelengths = [320, 400, 500, 600, 700, 800]
    ref_rad = mi.Float([mi.hosek_sun_rad(turb, wav, eta_ray, gamma) for wav in wavelengths])

    phi = dr.pi/5
    sun_theta = dr.pi/2 - eta_ray - gamma
    sun_theta = dr.select(sun_theta < 0, dr.pi/2 - eta_ray + gamma, sun_theta)
    plugin = make_emitter(turb=turb,
                          sun_phi=phi,
                          sun_theta=sun_theta,
                          albedo=0.0,
                          sun_scale=1.0,
                          sky_scale=0.0)

    # Generate rays
    si = mi.SurfaceInteraction3f()
    si.wavelengths = mi.Float(wavelengths)

    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(dr.pi/2 - eta_ray)
    si.wi = -mi.Vector3f(cp * st, sp * st, ct)

    # Evaluate the plugin
    res = plugin.eval(si)[0]
    rel_err = dr.abs(res - ref_rad) / (ref_rad + 1e-6)

    rtol = 1e-3
    assert rel_err <= rtol, (f"Fail when rendering sun with ray at elevation {eta_ray:.2f} and gamma {gamma:.2f}\n"
                             f"Mean relative error: {rel_err}, threshold: {rtol}")
