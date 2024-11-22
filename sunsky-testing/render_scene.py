import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_spectral")

dr.set_log_level(dr.LogLevel.Warn)
from rendering.spherical_sensor import SphericalSensor

def render_scene(scene_name):
    dr.print("Rendering test scene")

    scene = mi.load_file(f"sunsky-testing/res/scene/{scene_name}.xml")
    image = mi.render(scene, spp=128)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"sunsky-testing/res/renders/{scene_name}.exr", image)

render_scene("sky_rgb_3")