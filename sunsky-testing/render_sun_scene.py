import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

def render_scene(t, a, eta, phi_sun):
    dr.print("Rendering test scene {t}, {a}, {eta}", t=t, a=a, eta=dr.rad2deg(eta))

    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'direct'
        },
        'sensor': {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at([0, 0, 0], [cp * st, sp * st, ct], [0, 0, 1]),
            'fov': 1,
            'sampler': {
                'type': 'independent',
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
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

    scene = mi.load_dict(scene)
    return mi.render(scene, spp=512)



def render_and_write_scene(scene_name):
    t, a = 5, 0.5
    eta = dr.deg2rad(0.9)
    phi_sun = -4*dr.pi/5
    image = render_scene(t, a, eta, phi_sun)

    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)


if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")

    if mi.variant() == "cuda_ad_rgb":
        render_and_write_scene("test_sun_rgb")
    elif mi.variant() == "cuda_ad_spectral":
        render_and_write_scene("test_sun_spec")
    else:
        render_and_write_scene("test_sun_scalar")
