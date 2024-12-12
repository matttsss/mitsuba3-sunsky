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
                'type': 'independent',
                'sample_count': 1024
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 512,
            }
        },
        'emitter': {
            'type': 'sunsky',
            'sunDirection': [cp * st, sp * st, ct],
            'sunScale': 0.0,
            'turbidity': t,
            'albedo': a,
        }
    }

    scene = mi.load_dict(scene)
    return mi.render(scene, spp=1024)



def render_and_write_scene(scene_name):
    t, a = 2.5, 0.0
    eta = dr.deg2rad(22.5)
    phi_sun = -4*dr.pi/5
    image = render_scene(t, a, eta, phi_sun)

    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)

if __name__ == "__main__":
    mi.set_variant("llvm_rgb")
    dr.set_log_level(dr.LogLevel.Warn)
    render_and_write_scene("sky_rgb_3")
