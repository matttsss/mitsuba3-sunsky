import sys
sys.path.insert(0, "build/python")

import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")

def plot_pdf():
    t, a, eta = 6, 0.5, dr.deg2rad(50.2)
    phi_sun = -4*dr.pi/5

    sp_sun, cp_sun = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    sky = {
        "type": "sunsky",
        "sun_direction": [cp_sun * st, sp_sun * st, ct],
        "turbidity": t,
        "albedo": a
    }

    phi = dr.pi/4
    thetas = dr.linspace(mi.Float, 0, 2*dr.epsilon(mi.Float), 1024)
    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(thetas)

    rays = mi.Vector3f(cp*st, sp*st, ct)

    it = mi.Interaction3f()
    ds = mi.DirectionSample3f()
    ds.d = rays

    sky = mi.load_dict(sky)
    pdf = sky.pdf_direction(it, ds)

    dr.print(dr.asin(dr.epsilon(mi.Float)))

    plt.plot(thetas.numpy(), pdf.numpy())
    plt.show()

if __name__ == "__main__":
    plot_pdf()