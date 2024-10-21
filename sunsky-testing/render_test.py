import sys
sys.path.insert(0, "build/python")

import drjit as dr
dr.set_flag(dr.JitFlag.Debug, True)

import mitsuba as mi
mi.set_variant("llvm_rgb")


from rendering.sunsky_plugin import SunskyEmitter
from rendering.spherical_sensor import SphericalSensor

def render_scene(scene_name):
    dr.print("Rendering test scene")

    scene = mi.load_file(f"../scenes/{scene_name}.xml")
    image = mi.render(scene, spp=128)
    mi.util.write_bitmap(f"../renders/{scene_name}.png", image)
    mi.util.write_bitmap(f"../renders/{scene_name}.exr", image)

render_scene("sky_rgb")