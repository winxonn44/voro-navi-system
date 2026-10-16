"""Path smoothing: one post-process, applied identically to every planner.

Greedy line-of-sight shortcutting ("string pulling"). The raw path is first
resampled every 0.5 px, so grid paths and RRT*'s continuous polylines go
through exactly the same procedure. From the current anchor the next
waypoint is the farthest resampled point reachable by one straight segment,
extending one point at a time and stopping at the first segment that fails.

Two modes, both reading the planner-side fields (`safe`, the JFA clearance),
never the exact EDT that `path_metrics` grades with:
  safe:     a segment is valid if every sample (every 0.25 px, nearest
            pixel, the same rule as `segment_free` and `path_metrics`) is in
            `safe`, i.e. has clearance >= the robot radius. Shortest, but it
            gives clearance back: shortcuts pull the path towards corners.
  preserve: additionally, every sample's clearance must be >= the minimum
            clearance of the raw path between the segment's two ends. The
            smoothed path's minimum clearance is then never below the raw
            path's, up to the difference between the JFA and exact fields.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .metrics import resample

SMOOTH_MODES = ("safe", "preserve")
_SPACING = 0.25  # segment check spacing, as in planners.segment_free / metrics.resample


@njit(cache=True)
def _segment_min(clr, safe, ay, ax, by, bx):
    """Minimum clearance along a->b at the nearest pixel every 0.25 px;
    -1 if any sample leaves the grid or `safe`."""
    h, w = safe.shape
    n = max(2, int(math.ceil(math.hypot(by - ay, bx - ax) / 0.25)) + 1)
    lo = np.inf
    for k in range(n):
        t = k / (n - 1)
        y = int(np.rint(ay + t * (by - ay)))
        x = int(np.rint(ax + t * (bx - ax)))
        if y < 0 or y >= h or x < 0 or x >= w or not safe[y, x]:
            return -1.0
        c = clr[y, x]
        if c < lo:
            lo = c
    return lo


@njit(cache=True)
def _shortcut(pts, pt_clr, clr, safe, preserve):
    n = pts.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    i = 0
    while i < n - 1:
        j = i + 1
        raw_min = min(pt_clr[i], pt_clr[j])
        while j + 1 < n:
            nxt_min = min(raw_min, pt_clr[j + 1])
            m = _segment_min(clr, safe, pts[i, 0], pts[i, 1], pts[j + 1, 0], pts[j + 1, 1])
            if m < 0 or (preserve and m < nxt_min):
                break
            j += 1
            raw_min = nxt_min
        keep[j] = True
        i = j
    return keep


def smooth_path(path: np.ndarray, safe: np.ndarray, clearance: np.ndarray,
                mode: str = "safe") -> np.ndarray:
    """Shortcut `path` (float [N, 2] (y, x)); returns float [M, 2] with the
    same first and last point. See the module docstring for the modes."""
    if mode not in SMOOTH_MODES:
        raise ValueError(f"mode must be one of {SMOOTH_MODES}")
    path = np.asarray(path, dtype=np.float64)
    if len(path) <= 2:
        return path.copy()
    pts = resample(path, 0.5)
    h, w = safe.shape
    pix = np.clip(np.rint(pts).astype(np.int64), 0, [h - 1, w - 1])
    clr = np.ascontiguousarray(np.where(np.isfinite(clearance), clearance, 1e18), dtype=np.float64)
    pt_clr = clr[pix[:, 0], pix[:, 1]]
    keep = _shortcut(np.ascontiguousarray(pts), pt_clr, clr,
                     np.ascontiguousarray(safe, dtype=np.bool_), mode == "preserve")
    return pts[keep]
