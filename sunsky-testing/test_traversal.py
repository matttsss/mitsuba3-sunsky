import sys
sys.path.insert(0, "build/python")

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb_double")

from helpers import get_north_hemisphere_rays

def get_emitter(t, a, eta, phi_sun):
    sp, cp = dr.sincos(phi_sun)
    st, ct = dr.sincos(dr.pi/2 - eta)
    emitter = {
        'type': 'sunsky',
        'sun_direction': [cp * st, sp * st, ct],
        'turbidity': t,
        'albedo': a,
    }

    return mi.load_dict(emitter)

def get_scene(t, a, eta, phi_sun):
    from rendering.spherical_sensor import SphericalSensor

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
            'turbidity': t,
            'albedo': a,
        }
    }

    return mi.load_dict(scene)

def diff_render(scene, key):
    params = mi.traverse(scene)

    # Finite differences
    if False:
        d_v = 1e-6
        value = params[key]
        left_v = value - d_v
        right_v = value + d_v

        params[key] = left_v
        params.update()
        image_left = mi.render(scene, params, spp=128)

        params[key] = right_v
        params.update()
        image_right = mi.render(scene, params, spp=128)

        finite_diff = (image_right - image_left) / (right_v - left_v)

        finite_diff = mi.Bitmap(finite_diff).convert(component_format=mi.Struct.Type.Float32)

        mi.util.write_bitmap("sunsky-testing/res/renders/finite_diff.png", finite_diff)
        mi.util.write_bitmap("sunsky-testing/res/renders/finite_diff.exr", finite_diff)

    # Use AD framework
    dr.enable_grad(params[key])
    params.update()
    dr.graphviz_ad().view()

    image = mi.render(scene, params, spp=128)
    dr.forward(params[key])
    grad_image = dr.grad(image)

    grad_image = mi.Bitmap(grad_image).convert(component_format=mi.Struct.Type.Float32)

    mi.util.write_bitmap("sunsky-testing/res/renders/grad_image.png", grad_image)
    mi.util.write_bitmap("sunsky-testing/res/renders/grad_image.exr", grad_image)


def diff_plugin(plugin, key):
    params = mi.traverse(plugin)
    si = dr.zeros(mi.SurfaceInteraction3f)
    render_shape = (256, 1024)
    si.wi = get_north_hemisphere_rays(render_shape)

    # Finite differences
    if False:
        d_v = 1e-6
        value = params[key]
        left_v = value - d_v
        right_v = value + d_v

        params[key] = left_v
        params.update()
        image_left = dr.reshape(mi.TensorXf, plugin.eval(si), (*render_shape, 3))


        params[key] = right_v
        params.update()
        image_right = dr.reshape(mi.TensorXf, plugin.eval(si), (*render_shape, 3))

        finite_diff = (image_right - image_left) / (right_v - left_v)

        finite_diff = mi.Bitmap(finite_diff).convert(component_format=mi.Struct.Type.Float32)

        mi.util.write_bitmap("sunsky-testing/res/renders/finite_diff.png", finite_diff)
        mi.util.write_bitmap("sunsky-testing/res/renders/finite_diff.exr", finite_diff)

    # Use AD framework
    dr.enable_grad(params[key])
    params.update()

    image = dr.reshape(mi.TensorXf, plugin.eval(si), (*render_shape, 3)) if dr.hint(mi.is_rgb, mode="scalar") else plugin.eval(si)
    dr.set_label(image, "image")

    dr.forward_from(params[key], dr.ADFlag.ClearNone)
    dr.graphviz_ad().view()

    grad_image = dr.grad(image)

    if dr.hint(mi.is_rgb, mode="scalar"):
        grad_image = mi.Bitmap(grad_image).convert(component_format=mi.Struct.Type.Float32)

        mi.util.write_bitmap("sunsky-testing/res/renders/grad_image.png", grad_image)
        mi.util.write_bitmap("sunsky-testing/res/renders/grad_image.exr", grad_image)


if __name__ == "__main__":
    t, a, eta = 6, 0.5, dr.deg2rad(50)
    phi_sun = -4*dr.pi/5

    #diff_render(get_scene(t, a, eta, phi_sun), 'emitter.turbidity')
    diff_plugin(get_emitter(t, a, eta, phi_sun), 'turbidity')

