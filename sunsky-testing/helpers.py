import drjit as dr
import mitsuba as mi

def get_north_hemisphere_rays(resolution, ret_thetas=False):
    phi, thetas = dr.meshgrid(
        dr.linspace(mi.Float, -dr.pi, dr.pi, resolution[0]),
        dr.linspace(mi.Float, 0, dr.pi/2, resolution[1])
    )

    st, ct = dr.sincos(thetas)
    sp, cp = dr.sincos(phi)

    if ret_thetas:
        return mi.Vector3f(cp * st, sp * st, ct), thetas
    else:
        return mi.Vector3f(cp * st, sp * st, ct)

def get_camera_rays(cam_dir, apperture, image_res):
    cam_width, cam_height = apperture

    # Construct a grid of 2D coordinates
    x, y = dr.meshgrid(
        dr.linspace(mi.Float, -cam_width  / 2,   cam_width / 2, image_res[0]),
        dr.linspace(mi.Float, -cam_height / 2,  cam_height / 2, image_res[1])
    )

    # Ray origin in local coordinates
    ray_origin_local = mi.Vector3f(x, y, 0)

    # Ray origin in world coordinates
    return dr.normalize(mi.Frame3f(cam_dir).to_world(ray_origin_local))

