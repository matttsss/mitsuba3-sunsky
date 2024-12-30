import sys; sys.path.insert(0, "build/python")

import numpy as np
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

def get_dir(eta, phi):
    theta = dr.pi/2 - eta
    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(theta)

    return [cp * st, sp * st, ct]

def get_plugin(t, a, eta, phi):
    return mi.load_dict({
            'type': 'sunsky',
            'sun_direction': get_dir(eta, phi),
            'sun_scale': 0.0,
            'turbidity': t,
            'albedo': a,
        })


def plot_turb():
    t, a, eta, phi = 8, 0.0, dr.deg2rad(50), -4*dr.pi/5
    plugin = get_plugin(t, a, eta, phi)

    key = "turbidity"
    params = mi.traverse(plugin)

    rs = []
    gs = []
    bs = []

    turbidities = np.linspace(1, 10, 500)

    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -mi.Vector3f(get_dir(eta, phi + 0.2))

    for t in turbidities:
        params[key] = t
        params.update()

        color = plugin.eval(si)

        rs.append(color[0])
        gs.append(color[1])
        bs.append(color[2])

    plt.plot(turbidities, rs, color="red")
    plt.plot(turbidities, gs, color="green")
    plt.plot(turbidities, bs, color="blue")
    plt.ylim(bottom=0, top=1)
    plt.show()


if __name__ == "__main__":
    plot_turb()
