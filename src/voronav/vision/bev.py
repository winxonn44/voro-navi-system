"""Bird's-eye view: floor mask in the camera image -> metric occupancy grid.

Flat-ground model. The floor is the plane Z = 0 of a ground frame G whose
origin is the point on the floor directly below the camera: X forward (the
camera's heading projected onto the floor), Y left, Z up. The camera sits at
(0, 0, height_m), pitched down by `pitch_rad` and rolled by `roll_rad`
(positive = right side down). Pixels use the OpenCV convention: u right,
v down, the center of pixel (u, v) at (u, v).

A ground point maps to the image by the homography

    [u, v, 1]^T ~ H [X, Y, 1]^T,   H = K R^T [e1 | e2 | -C],   C = (0, 0, h)

where R holds the camera axes (right, down, forward) as columns in G. The
third homogeneous coordinate is the point's depth along the optical axis, so
w <= 0 means the point is behind the camera (beyond the horizon).

BEV grid layout (shared by every function here): the robot is at the bottom
center, forward is up. `extent_m = (forward_m, width_m)` covers
X in [0, forward_m] and Y in [-width_m/2, width_m/2]; row 0 is the far edge,
column 0 the left edge. Cell (r, c) has its center at

    X = forward_m - (r + 0.5) / px_per_m,   Y = width_m/2 - (c + 0.5) / px_per_m

Output convention (voronav.maps): True = obstacle OR unseen. A cell is free
only when every sample in it projects in front of the camera, inside the
image, within `max_range_m` of the robot, and onto a floor pixel. Unseen is
obstacle on purpose: Q3 found that floor leaking into walls causes
collisions while shrinking the floor only strands the planner.

Failure modes (see docs/Q4_DATA.md): anything that is not on the floor plane
is mis-placed (a table top at 0.7 m lands far behind the table; ramps and
stairs warp); a pitch or height error rescales the whole grid; and at range
one image pixel covers many BEV cells (the footprint grows ~ range^2 / (f h)),
so far cells are blocky and one mislabeled pixel blocks or frees a long
streak. `max_range_m` caps that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, for an image of width x height."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_rad: float) -> "Intrinsics":
        """Square pixels, principal point at the image center."""
        f = (width / 2) / np.tan(hfov_rad / 2)
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, width, height)


@dataclass(frozen=True)
class CameraPose:
    """Camera placement over a flat floor."""

    height_m: float
    pitch_rad: float = 0.0  # positive = looking down
    roll_rad: float = 0.0  # positive = right side down


def camera_rotation(cam: CameraPose) -> np.ndarray:
    """3x3 matrix whose columns are the camera's right, down and forward
    axes expressed in the ground frame G (X fwd, Y left, Z up)."""
    sp, cp = np.sin(cam.pitch_rad), np.cos(cam.pitch_rad)
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-sp, 0.0, -cp])
    fwd = np.array([cp, 0.0, -sp])
    sr, cr = np.sin(cam.roll_rad), np.cos(cam.roll_rad)
    right, down = cr * right + sr * down, -sr * right + cr * down
    return np.stack([right, down, fwd], axis=1)


def ground_to_image_homography(intr: Intrinsics, cam: CameraPose) -> np.ndarray:
    """H with [u, v, w] = H [X, Y, 1]; w > 0 is in front of the camera."""
    if cam.height_m <= 0:
        raise ValueError("camera height must be positive (camera above the floor)")
    R = camera_rotation(cam)
    M = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -cam.height_m]])
    return intr.K @ R.T @ M


def project_ground(H: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground points [..., 2] (m) -> pixel coords [..., 2] and w [...]."""
    xy = np.asarray(xy, dtype=float)
    p = xy @ H[:, :2].T + H[:, 2]
    w = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / w[..., None]
    return uv, w


def image_to_ground(intr: Intrinsics, cam: CameraPose, uv: np.ndarray) -> np.ndarray:
    """Pixels [..., 2] -> ground points [..., 2] (m); NaN above the horizon."""
    uv = np.asarray(uv, dtype=float)
    rays_c = np.stack(
        [(uv[..., 0] - intr.cx) / intr.fx, (uv[..., 1] - intr.cy) / intr.fy, np.ones(uv.shape[:-1])],
        axis=-1,
    )
    rays = rays_c @ camera_rotation(cam).T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rays[..., 2] < 0, cam.height_m / -rays[..., 2], np.nan)
    return np.stack([t * rays[..., 0], t * rays[..., 1]], axis=-1)


def blind_range_m(intr: Intrinsics, cam: CameraPose) -> float:
    """Distance ahead of the robot hidden below the image's bottom edge
    (center column). Cells closer than this are always unseen."""
    xy = image_to_ground(intr, cam, np.array([[intr.cx, intr.height - 0.5]]))[0]
    return float(xy[0]) if np.isfinite(xy[0]) else float("inf")


def bev_shape(px_per_m: float, extent_m: tuple[float, float]) -> tuple[int, int]:
    return int(round(extent_m[0] * px_per_m)), int(round(extent_m[1] * px_per_m))


def bev_cell_centers(px_per_m: float, extent_m: tuple[float, float], supersample: int = 1) -> np.ndarray:
    """Ground coords [h*s, w*s, 2] of the sample points (s x s per cell)."""
    h, w = bev_shape(px_per_m, extent_m)
    s = supersample
    step = 1.0 / (px_per_m * s)
    r = (np.arange(h * s) + 0.5) * step
    c = (np.arange(w * s) + 0.5) * step
    X = extent_m[0] - r
    Y = extent_m[1] / 2 - c
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    return np.stack([XX, YY], axis=-1)


def robot_cell(px_per_m: float, extent_m: tuple[float, float]) -> tuple[int, int]:
    """Grid cell (row, col) holding the robot's ground point (X=0, Y=0)."""
    h, w = bev_shape(px_per_m, extent_m)
    return h - 1, min(w - 1, int(extent_m[1] / 2 * px_per_m))


def bev_from_homography(
    mask: np.ndarray,
    H: np.ndarray,
    px_per_m: float,
    extent_m: tuple[float, float],
    max_range_m: float | None = None,
    supersample: int = 2,
) -> np.ndarray:
    """Occupancy grid from an image floor mask and a ground->image H.

    Nearest-pixel sampling at `supersample`^2 points per cell; a cell is free
    only if all of its samples are free (conservative)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be [H, W]")
    ih, iw = mask.shape
    xy = bev_cell_centers(px_per_m, extent_m, supersample)
    uv, w = project_ground(H, xy)
    ui = np.floor(uv[..., 0] + 0.5)
    vi = np.floor(uv[..., 1] + 0.5)
    ok = (w > 0) & np.isfinite(ui) & np.isfinite(vi) & (ui >= 0) & (ui < iw) & (vi >= 0) & (vi < ih)
    if max_range_m is not None:
        ok &= np.hypot(xy[..., 0], xy[..., 1]) <= max_range_m
    free = np.zeros(ok.shape, dtype=bool)
    free[ok] = mask[vi[ok].astype(np.intp), ui[ok].astype(np.intp)]
    h, wd = bev_shape(px_per_m, extent_m)
    s = supersample
    free = free.reshape(h, s, wd, s).all(axis=(1, 3))
    return ~free


# Defaults are the recommended planning grid for the Q4 data (docs/Q4_DATA.md):
# 5 cm cells, 5 m ahead x 8 m wide, range capped at 5 m.


def mask_to_bev(
    mask: np.ndarray,
    intrinsics: Intrinsics,
    cam: CameraPose,
    px_per_m: float = 20.0,
    extent_m: tuple[float, float] = (5.0, 8.0),
    max_range_m: float | None = 5.0,
    supersample: int = 2,
) -> np.ndarray:
    """bool [h, w] occupancy (True = obstacle or unseen) from a floor mask."""
    if mask.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"mask {mask.shape} does not match intrinsics {intrinsics.height}x{intrinsics.width}")
    H = ground_to_image_homography(intrinsics, cam)
    return bev_from_homography(mask, H, px_per_m, extent_m, max_range_m, supersample)


def fov_mask(
    intrinsics: Intrinsics,
    cam: CameraPose,
    px_per_m: float = 20.0,
    extent_m: tuple[float, float] = (5.0, 8.0),
    max_range_m: float | None = 5.0,
) -> np.ndarray:
    """bool [h, w], True where the cell is visible (in the camera's view
    and range). An all-floor mask through mask_to_bev gives ~this."""
    full = np.ones((intrinsics.height, intrinsics.width), dtype=bool)
    return ~mask_to_bev(full, intrinsics, cam, px_per_m, extent_m, max_range_m, supersample=1)


# ---------------------------------------------------------------------------
# Calibration from marked floor points
# ---------------------------------------------------------------------------


def homography_from_points(ground_xy: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
    """Ground->image H from >= 4 correspondences (normalized DLT).

    ground_xy are floor points in the BEV ground frame (m, X fwd, Y left);
    image_uv the pixels where they appear. The sign of H is fixed so that
    w > 0 at the given points (they are in front of the camera)."""
    g = np.asarray(ground_xy, dtype=float)
    p = np.asarray(image_uv, dtype=float)
    if g.shape != p.shape or g.ndim != 2 or g.shape[1] != 2 or len(g) < 4:
        raise ValueError("need >= 4 matching (x, y) / (u, v) points")

    def normalizer(pts):
        m = pts.mean(axis=0)
        d = np.sqrt(((pts - m) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / d if d > 0 else 1.0
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])

    Tg, Tp = normalizer(g), normalizer(p)
    gn = np.c_[g, np.ones(len(g))] @ Tg.T
    pn = np.c_[p, np.ones(len(p))] @ Tp.T
    rows = []
    for (x, y, _), (u, v, _) in zip(gn, pn):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(rows)
    _, sv, vt = np.linalg.svd(A)
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError("degenerate points (three or more collinear?)")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tp) @ Hn @ Tg
    w = np.c_[g, np.ones(len(g))] @ H[2]
    if np.all(w < 0):
        H = -H
    elif not np.all(w > 0):
        raise ValueError("points straddle the horizon")
    return H / np.abs(H).max()


def bev_from_points(
    mask: np.ndarray,
    ground_xy: np.ndarray,
    image_uv: np.ndarray,
    px_per_m: float = 20.0,
    extent_m: tuple[float, float] = (5.0, 8.0),
    max_range_m: float | None = 5.0,
    supersample: int = 2,
) -> np.ndarray:
    """mask_to_bev, calibrated from 4+ marked floor points instead of
    intrinsics + height + pitch (e.g. tape marks at known spots on the floor)."""
    H = homography_from_points(ground_xy, image_uv)
    return bev_from_homography(mask, H, px_per_m, extent_m, max_range_m, supersample)
