import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

plugin_name = "sunsky"

def check_chi2():
    t, a = 6, 0.5
    eta = dr.deg2rad(50.2)
    phi_sun = -4*dr.pi/5

    sp_sun, cp_sun = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)

    # Compute coefficients
    sky = {
        "type": plugin_name,
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "sun_scale": 0.0,
        "turbidity": t,
        "albedo": a
    }
    sample_func, pdf_func = mi.chi2.EmitterAdapter(plugin_name, sky)

    test = mi.chi2.ChiSquareTest(
        domain=mi.chi2.SphericalDomain(),
        pdf_func= pdf_func,
        sample_func= sample_func,
        sample_dim=2,
        sample_count=1000000000,
        res=216,
        ires=32
    )

    assert test.run()

def plot_pdf():
    a, t, eta = 0.5, 6.5, dr.deg2rad(45)
    side = 512
    render_shape = (side//4, side)

    phi_sun = dr.pi / 2
    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)

    # Compute coefficients
    sky = mi.load_dict({
        "type": plugin_name,
        "sun_direction": [cp * st, sp * st, ct],
        "sun_scale": 0.0,
        "turbidity": t,
        "albedo": a
    })

    si = dr.zeros(mi.SurfaceInteraction3f)
    it = dr.zeros(mi.Interaction3f)
    ds = dr.zeros(mi.DirectionSample3f)
    hemisphere_dir, (_, thetas) = get_north_hemisphere_rays(render_shape, True, 0.15)

    nb_lines = 0
    idx = nb_lines * render_shape[1] + dr.arange(mi.UInt32, render_shape[0] * render_shape[1] - nb_lines * render_shape[1])

    # ================ Colored -> PDF ==================
    temp_shape = (render_shape[0] * 2, render_shape[1])
    si.wi = -get_spherical_rays(temp_shape)

    color_render = dr.reshape(mi.TensorXf, sky.eval(si), (*temp_shape, 3))

    envmap = mi.load_dict({
        "type": "envmap",
        "bitmap": mi.Bitmap(color_render)
    })

    ds.d = mi.Vector3f(hemisphere_dir.y, hemisphere_dir.z, -hemisphere_dir.x)

    pdf_ref = envmap.pdf_direction(it, ds)
    pdf_ref = dr.gather(mi.Float, pdf_ref, idx)

    # ==================     PDF     ====================
    ds.d = hemisphere_dir
    pdf_render = sky.pdf_direction(it, ds)
    pdf_render = dr.gather(mi.Float, pdf_render, idx)

    # ================  RELATIVE ERROR  ==================
    relative_error = dr.abs((pdf_ref - pdf_render) / (pdf_ref + 0.01))

    pdf_ref = dr.reshape(mi.TensorXf, pdf_ref, (render_shape[0]-nb_lines, render_shape[1]))
    pdf_render = dr.reshape(mi.TensorXf, pdf_render, (render_shape[0]-nb_lines, render_shape[1]))
    relative_error = dr.reshape(mi.TensorXf, relative_error, (render_shape[0]-nb_lines, render_shape[1]))

    fig, axes = plt.subplots(nrows=3)

    vmin = dr.ravel(dr.min(relative_error))[0]
    vmax = dr.ravel(dr.max(relative_error))[0]

    ref_plot = axes[0].imshow(pdf_ref, vmin=0, interpolation='nearest', cmap='grey')
    hist_plot = axes[1].imshow(pdf_render, vmin=0, interpolation='nearest', cmap='grey')
    diff_plot = axes[2].imshow(relative_error, vmin=vmin, vmax=vmax, interpolation='nearest', cmap='coolwarm')

    # Add color bars
    fig.colorbar(ref_plot, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(hist_plot, ax=axes[1], fraction=0.046, pad=0.02)
    fig.colorbar(diff_plot, ax=axes[2], fraction=0.046, pad=0.02)

    axes[0].axis('off')
    axes[0].set_title("Bitmap PDF")

    axes[1].axis('off')
    axes[1].set_title("tGMM PDF")

    axes[2].axis('off')
    axes[2].set_title("Relative error")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    from helpers import get_north_hemisphere_rays, get_spherical_rays

    plot_pdf()
    #check_chi2()
