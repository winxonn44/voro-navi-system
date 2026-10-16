import numpy as np
import pytest

from voronav.vision.bev import (
    CameraPose,
    Intrinsics,
    bev_cell_centers,
    bev_from_points,
    bev_shape,
    blind_range_m,
    camera_rotation,
    fov_mask,
    ground_to_image_homography,
    homography_from_points,
    image_to_ground,
    mask_to_bev,
    project_ground,
    robot_cell,
)

INTR = Intrinsics.from_fov(160, 120, np.radians(90))


def render_floor_mask(intr, cam, inside):
    """Ray-cast every pixel onto the floor; True where inside(x, y)."""
    vv, uu = np.mgrid[0 : intr.height, 0 : intr.width]
    xy = image_to_ground(intr, cam, np.stack([uu, vv], axis=-1).astype(float))
    ok = np.isfinite(xy[..., 0])
    out = np.zeros(ok.shape, dtype=bool)
    out[ok] = inside(xy[ok, 0], xy[ok, 1])
    return out


def test_rotation_is_orthonormal_right_handed():
    for cam in [CameraPose(1.0), CameraPose(1.0, 0.4, 0.2), CameraPose(0.3, -0.1, -0.3)]:
        R = camera_rotation(cam)
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_level_camera_known_points():
    # Level camera, h = 1, 90 deg FOV: a floor point at distance X straight
    # ahead projects to v = cy + f h / X; lateral Y left goes to u = cx - f Y / X.
    cam = CameraPose(1.0)
    H = ground_to_image_homography(INTR, cam)
    f = INTR.fx
    uv, w = project_ground(H, np.array([[2.0, 0.0], [4.0, 1.0], [1.0, -0.5]]))
    assert np.all(w > 0)
    assert np.allclose(uv[0], [INTR.cx, INTR.cy + f / 2])
    assert np.allclose(uv[1], [INTR.cx - f / 4, INTR.cy + f / 4])
    assert np.allclose(uv[2], [INTR.cx + f / 2, INTR.cy + f])
    # behind the camera: w <= 0
    _, w = project_ground(H, np.array([[-1.0, 0.0]]))
    assert w[0] < 0


@pytest.mark.parametrize("pitch,roll", [(0.0, 0.0), (0.3, 0.0), (0.5, 0.1), (-0.1, -0.2), (1.2, 0.0)])
def test_homography_inverts_ray_casting(pitch, roll):
    cam = CameraPose(0.8, pitch, roll)
    H = ground_to_image_homography(INTR, cam)
    rng = np.random.default_rng(0)
    uv = np.c_[rng.uniform(0, INTR.width, 200), rng.uniform(0, INTR.height, 200)]
    xy = image_to_ground(INTR, cam, uv)
    hit = np.isfinite(xy[:, 0])
    assert hit.sum() > 20
    back, w = project_ground(H, xy[hit])
    assert np.all(w > 0)
    assert np.allclose(back, uv[hit], atol=1e-6)


def test_pitch_down_looks_at_point():
    # Pitched down by atan(h / d): the optical axis hits the floor at X = d.
    h, d = 1.2, 3.0
    cam = CameraPose(h, np.arctan2(h, d))
    xy = image_to_ground(INTR, cam, np.array([[INTR.cx, INTR.cy]]))[0]
    assert np.allclose(xy, [d, 0.0])


def test_positive_roll_is_right_side_down():
    # Right side down: the right of the image looks further down. Just below
    # the center row, the right edge hits the floor and the left edge is
    # still above the horizon.
    cam = CameraPose(1.0, 0.0, 0.2)
    xy_l = image_to_ground(INTR, cam, np.array([[0.0, INTR.cy + 5]]))[0]
    xy_r = image_to_ground(INTR, cam, np.array([[INTR.width - 1.0, INTR.cy + 5]]))[0]
    assert np.isfinite(xy_r).all() and xy_r[1] < 0
    assert np.isnan(xy_l).all()


def test_bad_height_rejected():
    with pytest.raises(ValueError):
        ground_to_image_homography(INTR, CameraPose(0.0))
    with pytest.raises(ValueError):
        ground_to_image_homography(INTR, CameraPose(-1.0, 0.2))


def test_looking_up_sees_no_floor():
    # Pitched up past half the vertical FOV: no pixel reaches the floor.
    cam = CameraPose(1.0, -np.radians(80))
    assert not fov_mask(INTR, cam).any()
    assert blind_range_m(INTR, cam) == float("inf")


def test_grid_layout():
    px, ext = 10.0, (4.0, 6.0)
    assert bev_shape(px, ext) == (40, 60)
    c = bev_cell_centers(px, ext)
    assert np.allclose(c[0, 0], [3.95, 2.95])  # far-left
    assert np.allclose(c[-1, -1], [0.05, -2.95])  # near-right
    assert robot_cell(px, ext) == (39, 30)


@pytest.mark.parametrize("cam", [CameraPose(1.0, 0.35), CameraPose(0.5, 0.15), CameraPose(1.5, 0.6, 0.05)])
def test_rendered_rectangle_maps_back(cam):
    # A floor rectangle rendered into the image comes back as the same metric
    # rectangle in the BEV, up to a cell or two at its edges.
    x0, x1, y0, y1 = 1.5, 3.5, -0.8, 0.6
    intr = Intrinsics.from_fov(320, 240, np.radians(90))
    mask = render_floor_mask(intr, cam, lambda x, y: (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
    px, ext = 20.0, (5.0, 4.0)
    occ = mask_to_bev(mask, intr, cam, px, ext, max_range_m=None, supersample=1)
    xy = bev_cell_centers(px, ext)
    truth = (xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= y0) & (xy[..., 1] <= y1)
    free = ~occ
    # nothing outside a 2-cell margin leaks in; the interior is free
    margin = 2.0 / px
    loose = (xy[..., 0] >= x0 - margin) & (xy[..., 0] <= x1 + margin)
    loose &= (xy[..., 1] >= y0 - margin) & (xy[..., 1] <= y1 + margin)
    tight = (xy[..., 0] >= x0 + margin) & (xy[..., 0] <= x1 - margin)
    tight &= (xy[..., 1] >= y0 + margin) & (xy[..., 1] <= y1 - margin)
    assert not (free & ~loose).any()
    assert free[tight].all()
    iou = (free & truth).sum() / (free | truth).sum()
    assert iou > 0.9


def test_fov_mask_wedge():
    # Level camera, 90 deg HFOV: visible cells are |Y| < X (the wedge), beyond
    # the blind range, within max range.
    cam = CameraPose(1.0, 0.0)
    intr = Intrinsics.from_fov(200, 200, np.radians(90))
    px, ext, rmax = 10.0, (6.0, 8.0), 5.0
    vis = fov_mask(intr, cam, px, ext, rmax)
    xy = bev_cell_centers(px, ext)
    X, Y = xy[..., 0], xy[..., 1]
    r = np.hypot(X, Y)
    blind = blind_range_m(intr, cam)
    assert np.isclose(blind, 1.0, atol=0.02)  # 45 deg down at the bottom edge
    clearly_in = (np.abs(Y) < X - 0.15) & (X > blind + 0.15) & (r < rmax - 0.15)
    clearly_out = (np.abs(Y) > X + 0.15) | (X < blind - 0.15) | (r > rmax + 0.15)
    assert vis[clearly_in].all()
    assert not vis[clearly_out].any()
    # the robot's own cell is never visible (it's below the camera)
    assert not vis[robot_cell(px, ext)]


def test_unseen_and_nonfloor_are_obstacles():
    cam = CameraPose(1.0, 0.3)
    none = np.zeros((INTR.height, INTR.width), dtype=bool)
    assert mask_to_bev(none, INTR, cam).all()
    full = ~none
    occ = mask_to_bev(full, INTR, cam, max_range_m=3.0)
    xy = bev_cell_centers(20.0, (5.0, 8.0))
    assert occ[np.hypot(xy[..., 0], xy[..., 1]) > 3.05].all()
    assert (~occ).any()


def test_supersample_is_conservative():
    cam = CameraPose(1.0, 0.3)
    rng = np.random.default_rng(1)
    mask = rng.random((INTR.height, INTR.width)) < 0.8
    a = mask_to_bev(mask, INTR, cam, supersample=1)
    b = mask_to_bev(mask, INTR, cam, supersample=3)
    assert b.dtype == bool and b.shape == a.shape
    assert b.sum() >= a.sum()


def test_mask_shape_checked():
    with pytest.raises(ValueError):
        mask_to_bev(np.ones((10, 10), bool), INTR, CameraPose(1.0))


def test_homography_from_points_matches_analytic():
    cam = CameraPose(0.9, 0.25, 0.03)
    H = ground_to_image_homography(INTR, cam)
    g = np.array([[1.5, -0.7], [1.5, 0.7], [4.0, -1.5], [4.0, 1.5]])
    uv, _ = project_ground(H, g)
    H2 = homography_from_points(g, uv)
    test = np.array([[2.0, 0.3], [3.3, -1.0], [1.2, 0.0]])
    a, _ = project_ground(H, test)
    b, w = project_ground(H2, test)
    assert np.all(w > 0)
    assert np.allclose(a, b, atol=1e-6)
    # a point behind the camera still gets w < 0 after the sign fix
    _, w = project_ground(H2, np.array([[-2.0, 0.0]]))
    assert w[0] < 0


def test_bev_from_points_equals_mask_to_bev():
    cam = CameraPose(1.1, 0.3)
    mask = render_floor_mask(INTR, cam, lambda x, y: (x > 1.8) & (np.abs(y) < 1.0))
    H = ground_to_image_homography(INTR, cam)
    g = np.array([[2.0, -1.0], [2.0, 1.0], [5.0, -1.0], [5.0, 1.0]])
    uv, _ = project_ground(H, g)
    a = mask_to_bev(mask, INTR, cam)
    b = bev_from_points(mask, g, uv)
    assert (a != b).mean() < 0.002  # float noise at cell boundaries only


def test_homography_from_points_degenerate():
    g = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    with pytest.raises(ValueError):
        homography_from_points(g, g * 10)
    with pytest.raises(ValueError):
        homography_from_points(g[:3], g[:3])
