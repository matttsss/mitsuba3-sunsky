import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

def render_scene(t, a, eta, phi_sun):
    from rendering.spherical_sensor import SphericalSensor

    dr.print("Rendering test scene")

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
            'sun_scale': 0.0,
            'sky_scale': 1.0,
            'turbidity': t,
            'albedo': a,
        }
    }

    scene = mi.load_dict(scene)
    return mi.render(scene, spp=2048)



def render_and_write_scene(scene_name):
    t, a = 3.2, 0.0
    eta = dr.deg2rad(75)
    phi_sun = -4*dr.pi/5
    image = render_scene(t, a, eta, phi_sun)

    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    dr.set_log_level(dr.LogLevel.Warn)
    mi.write_sun_sky_model_data("resources/sunsky/")

    if mi.variant() == "cuda_ad_rgb":
        render_and_write_scene("test_sun_rgb")
    elif mi.variant() == "cuda_ad_spectral":
        render_and_write_scene("test_sun_spec")
    else:
        render_and_write_scene("test_sun_scalar")