import sys
sys.path.insert(0, "build/python")

import drjit as dr
dr.set_flag(dr.JitFlag.Debug, True)

import mitsuba as mi
mi.set_variant("llvm_rgb")


from sunsky_plugin import SunskyEmitter

mi.register_emitter("constant_emitter", SunskyEmitter)

def render():
    dr.print("Rendering test scene")

    scene = mi.load_file("sunsky-testing/res/scene/simple.xml")
    image = mi.render(scene, spp=128)
    mi.util.write_bitmap("sunsky-testing/test.png", image)
    mi.util.write_bitmap("sunsky-testing/test.exr", image)

render()