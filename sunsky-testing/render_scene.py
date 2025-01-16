import sys; sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

    scene = mi.load_file("sunsky-testing/scenes/dragon/Shader_Dragon.xml")
    image = mi.render(scene, spp=1024)

    mi.util.write_bitmap("sunsky-testing/res/renders/dragon.exr", image)
    mi.util.write_bitmap("sunsky-testing/res/renders/dragon.png", image)
