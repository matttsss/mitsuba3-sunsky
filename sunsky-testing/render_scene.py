import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_spectral")

dr.set_log_level(dr.LogLevel.Warn)
from rendering.spherical_sensor import SphericalSensor

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
            'type': 'spherical',
            'sampler': {
                'type': 'independent',
                'sample_count': 128
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 512
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
    return mi.render(scene, spp=128)



def render_and_write_scene(scene_name):
    t, a = 1.2, 0.5
    eta = dr.deg2rad(80)
    phi_sun = -4*dr.pi/5
    image = render_scene(t, a, eta, phi_sun)

    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)

if __name__ == "__main__":
    render_and_write_scene("sky_rgb_3")
