import drjit as dr
import mitsuba as mi

def to_spherical(phi: mi.Float, theta: mi.Float) -> mi.Vector3f:
    sp, cp = dr.sincos(phi)
    st, ct = dr.sincos(theta)
    return mi.Vector3f(cp * st, sp * st, ct)

def from_spherical(v: mi.Vector3f) -> mi.Vector2f:
    """Returns (phi, theta)"""
    return mi.Vector2f(dr.atan2(v[1], v[0]), dr.acos(v[2]))


def get_north_hemisphere_rays(resolution, ret_angles = False):
    phi, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.two_pi, resolution[1]),
        dr.linspace(mi.Float, 0, dr.pi/2, resolution[0])
    )

    if ret_angles:
        return to_spherical(phi, thetas), (phi, thetas)
    else:
        return to_spherical(phi, thetas)

def get_spherical_rays(resolution, ret_angles = False):
    phi, thetas = dr.meshgrid(
        dr.linspace(mi.Float, 0, dr.two_pi, resolution[1]),
        dr.linspace(mi.Float, 0, dr.pi, resolution[0])
    )

    if ret_angles:
        return to_spherical(phi, thetas), (phi, thetas)
    else:
        return to_spherical(phi, thetas)

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

