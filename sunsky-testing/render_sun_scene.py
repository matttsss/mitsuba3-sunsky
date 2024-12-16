import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

def render_scene(t, a, eta, phi_sun):
    dr.print("Rendering test scene")

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
            'sunDirection': [cp * st, sp * st, ct],
            'turbidity': t,
            'albedo': a,
        }
    }

    scene = mi.load_dict(scene)
    return mi.render(scene, spp=512)



def render_and_write_scene(scene_name):
    t, a = 8, 0.5
    eta = dr.deg2rad(0.9)
    phi_sun = -4*dr.pi/5
    image = render_scene(t, a, eta, phi_sun)

    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)

if __name__ == "__main__":
    mi.set_variant("cuda_spectral")
    dr.set_log_level(dr.LogLevel.Warn)
    mi.write_sun_sky_model_data("sunsky-testing/res/datasets/ssm_dataset")

    if mi.variant() == "cuda_rgb":
        render_and_write_scene("test_sun_rgb")
    elif mi.variant() == "cuda_spectral":
        render_and_write_scene("test_sun_spec")
    else:
        render_and_write_scene("test_sun_scalar")
