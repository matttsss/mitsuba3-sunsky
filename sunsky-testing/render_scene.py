import sys; sys.path.insert(0, "build/python")

import mitsuba as mi

if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")

    scene = mi.load_file("sunsky-testing/res/scene/dragon/Shader_Dragon.xml")
    image = mi.render(scene, spp=64)
    image = mi.Bitmap(image).convert(component_format=mi.Struct.Type.Float32)


    mi.util.write_bitmap("sunsky-testing/res/renders/dragon.exr", image)
    mi.util.write_bitmap("sunsky-testing/res/renders/dragon.png", image)
