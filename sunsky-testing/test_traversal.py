import sys; sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")


def get_scene(t, a, eta, phi_sun):
    from rendering.spherical_sensor import SphericalSensor

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'direct'
        },
        'sensor': {
            'type': 'spherical',
            'sampler': {
                'type': 'independent'
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 512,
            }
        },
        'emitter': {
            'type': 'sunsky',
            'sun_direction': [cp * st, sp * st, ct],
            'turbidity': t,
            'albedo': a,
        }
    }

    return mi.load_dict(scene)

if __name__ == "__main__":
    t, a, eta = 3.2, 0.0, dr.deg2rad(75)
    phi_sun = -4*dr.pi/5

    scene = get_scene(t, a, eta, phi_sun)
    params = mi.traverse(scene)

    key = 'emitter.turbidity'
    dr.enable_grad(params[key])
    params.update()

    image = mi.render(scene, params, spp=128)
    dr.forward(params[key])

    # Fetch the image gradient values
    grad_image = dr.grad(image)

    import matplotlib.pyplot as plt
    plt.imshow(grad_image * 2.0)
    plt.axis('off')
    plt.show()
