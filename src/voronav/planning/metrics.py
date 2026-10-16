"""Path quality metrics, measured against an *independent* clearance field.

Clearance comes from SciPy's exact Euclidean distance transform of the
occupancy grid, not from the JFA field the Voronoi planner was built on, so
the planner under test cannot grade its own homework.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def exact_clearance(occ: np.ndarray) -> np.ndarray:
    """Distance (px) from each pixel center to the nearest obstacle pixel
    center; 0 on obstacles, inf when there are no obstacles."""
    if not occ.any():
        return np.full(occ.shape, np.inf)
    return distance_transform_edt(~occ)


def resample(path: np.ndarray, spacing: float = 0.25) -> np.ndarray:
    """Points every `spacing` px along the polyline (endpoints included)."""
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing)))
        t = np.arange(1, n + 1)[:, None] / n
        pts.extend(a + t * (b - a))
    return np.asarray(pts)


def path_metrics(path: np.ndarray, clearance: np.ndarray, safe: np.ndarray) -> dict:
    """length, clearance stats along the path, smoothness, collision check."""
    seg = np.diff(path, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    pts = np.rint(resample(path)).astype(np.int64)
    h, w = clearance.shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)
    pts_in = pts[inside]
    c = clearance[pts_in[:, 0], pts_in[:, 1]]
    # total heading change, ignoring zero-length segments
    moving = seg[seg_len > 1e-9]
    headings = np.arctan2(moving[:, 0], moving[:, 1])
    turn = np.abs((np.diff(headings) + math.pi) % (2 * math.pi) - math.pi)
    return {
        "length_px": float(seg_len.sum()),
        "min_clearance_px": float(c.min()) if inside.all() else 0.0,
        "mean_clearance_px": float(c.mean()),
        "p10_clearance_px": float(np.percentile(c, 10)),
        "total_turn_deg": float(np.degrees(turn.sum())),
        "collision_free": bool(inside.all() and safe[pts_in[:, 0], pts_in[:, 1]].all()),
        "n_waypoints": int(len(path)),
    }
