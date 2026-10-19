"""Ego-motion alignment of BEV grids, and Q3's temporal filter in an aligned frame.

Every BEV grid (voronav.vision.bev) is drawn in its own frame's ground
frame: robot at the bottom centre, forward up. When the camera moves, cell
(r, c) of frame j and cell (r, c) of frame t are different places on the
floor, so Q3's majority vote (voronav.temporal) cannot be applied to raw
grids. Here earlier grids are warped into the current frame with the
relative ground-plane pose (a 2D rigid transform) before voting.

The pose comes from the dataset (TartanGround's simulated, exact poses). A
real robot would use wheel or visual odometry instead, whose drift adds a
misalignment this module does not model.

Ground pose of a frame: (x, y, yaw) of the camera's floor point in a world
frame (x north, y east for TartanGround's NED poses; yaw from north towards
east), heading = the camera's forward axis projected on the floor.

The filter only ever changes cells the current frame can see (its field of
view): the output is graded against the current frame's ground truth, in
which everything out of view is unseen = obstacle. Memory of cells that
left the view (the blind zone right in front of the robot) would be
mapping, not filtering, and is not attempted.

Gates, applied to each new frame before it may join the vote:
  "iou"   Q3's gate: free-space IoU with the last accepted grid (aligned,
          over the cells both frames see) below `outlier_iou` -> reject.
  "leak"  the gate Q3 recommended, because the IoU gate is blind to leaks
          (a 2-3 px leak barely moves the IoU): count the cells the new
          frame marks free that the last accepted grid had as obstacle
          within `band_px` of its floor boundary, i.e. floor growing into
          walls. Reject when that is more than `leak_frac` of the last
          accepted grid's free area (over the shared view).
  "both"  reject when either gate fires.
After `patience` rejections in a row the current frame is accepted and the
history restarts from it (as in temporal.temporal_filter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from .bev import bev_shape
from .dataset import quat_to_matrix

GATES = ("none", "iou", "leak", "both")


def ground_poses(poses: np.ndarray) -> np.ndarray:
    """[N, 7] (x y z qx qy qz qw, NED, body x forward) -> [N, 3] (x, y, yaw)."""
    poses = np.asarray(poses, dtype=float)
    R = quat_to_matrix(poses[:, 3:])
    fwd = R[:, :, 0]  # body x axis in world
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    return np.stack([poses[:, 0], poses[:, 1], yaw], axis=1)


def _axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward and left unit vectors in the world (x north, y east, z down):
    left of north is west, (0, -1)."""
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([np.sin(yaw), -np.cos(yaw)])
    return f, left


def relative_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) mapping ground coords (X fwd, Y left, m) of frame `dst` to
    those of frame `src`: p_src = A @ p_dst + b. Poses are (x, y, yaw)."""
    fs, ls = _axes(src[2])
    fd, ld = _axes(dst[2])
    Bs = np.stack([fs, ls], axis=1)  # columns: src axes in world
    Bd = np.stack([fd, ld], axis=1)
    A = Bs.T @ Bd
    b = Bs.T @ (np.asarray(dst[:2]) - np.asarray(src[:2]))
    return A, b


def warp_bev(
    grid: np.ndarray,
    src_pose: np.ndarray,
    dst_pose: np.ndarray,
    px_per_m: float,
    extent_m: tuple[float, float],
    fill: bool = True,
) -> np.ndarray:
    """Resample `grid` (drawn in frame src) into frame dst's BEV layout.

    Nearest cell; cells of dst that fall outside src's grid get `fill`
    (True = obstacle/unseen for occupancy grids, False for visibility)."""
    grid = np.asarray(grid, dtype=bool)
    h, w = bev_shape(px_per_m, extent_m)
    if grid.shape != (h, w):
        raise ValueError(f"grid {grid.shape} does not match the BEV layout {(h, w)}")
    A, b = relative_transform(np.asarray(src_pose, float), np.asarray(dst_pose, float))
    r = np.arange(h) + 0.5
    c = np.arange(w) + 0.5
    X = extent_m[0] - r / px_per_m
    Y = extent_m[1] / 2 - c / px_per_m
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    Xs = A[0, 0] * XX + A[0, 1] * YY + b[0]
    Ys = A[1, 0] * XX + A[1, 1] * YY + b[1]
    # inverse of the cell-centre formula, rounded to the nearest cell
    rs = np.floor((extent_m[0] - Xs) * px_per_m + 1e-9).astype(np.int64)
    cs = np.floor((extent_m[1] / 2 - Ys) * px_per_m + 1e-9).astype(np.int64)
    ok = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
    out = np.full((h, w), fill, dtype=bool)
    out[ok] = grid[rs[ok], cs[ok]]
    return out


def aligned_iou(cur: np.ndarray, prev: np.ndarray, both_seen: np.ndarray) -> float:
    """Free-space IoU of two aligned occupancy grids over the cells both see."""
    fc, fp = ~cur & both_seen, ~prev & both_seen
    union = int((fc | fp).sum())
    return int((fc & fp).sum()) / union if union else 1.0


def leak_score(cur: np.ndarray, prev: np.ndarray, both_seen: np.ndarray, band_px: float) -> float:
    """New free space near the previous floor boundary, relative to the
    previous free area: cells `cur` marks free that `prev` marks obstacle
    within `band_px` of prev's floor, over the cells both frames see."""
    prev_free = ~prev & both_seen
    n_prev = int(prev_free.sum())
    if n_prev == 0:
        return 0.0
    if prev_free.all():
        return 0.0
    d = distance_transform_edt(~prev_free)  # distance to prev's floor
    band = prev & both_seen & (d <= band_px)
    return int((~cur & band).sum()) / n_prev


@dataclass
class FilterStep:
    """What the filter did on one frame."""

    gate: str                 # first | accept | reject | reset
    iou: float | None         # aligned IoU with the last accepted grid
    leak: float | None        # leak_score against the last accepted grid
    history: list[int] = field(default_factory=list)  # frames in the vote


class AlignedTemporalFilter:
    """Causal gate + majority vote over the last `window` accepted grids,
    each warped into the current frame. Feed frames in order with step()."""

    def __init__(
        self,
        px_per_m: float,
        extent_m: tuple[float, float],
        window: int = 5,
        gate: str = "iou",
        outlier_iou: float = 0.8,
        leak_frac: float = 0.05,
        band_px: float = 3.0,
        patience: int | None = 3,
    ):
        if window < 1:
            raise ValueError("window must be >= 1")
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        self.px_per_m, self.extent_m = px_per_m, tuple(extent_m)
        self.window, self.gate = window, gate
        self.outlier_iou, self.leak_frac, self.band_px = outlier_iou, leak_frac, band_px
        self.patience = patience
        self.hist: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []  # (t, occ, seen, pose)
        self.rejected = 0
        self.t = 0

    def _warp(self, grid, src, dst, fill):
        return warp_bev(grid, src, dst, self.px_per_m, self.extent_m, fill=fill)

    def step(self, occ: np.ndarray, seen: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, FilterStep]:
        """occ: this frame's BEV (True = obstacle/unseen); seen: its field of
        view (True = visible); pose: its ground pose (x, y, yaw)."""
        t = self.t
        self.t += 1
        pose = np.asarray(pose, dtype=float)
        iou = leak = None
        if not self.hist:
            self.hist.append((t, occ, seen, pose))
            gate = "first"
        else:
            _, p_occ, p_seen, p_pose = self.hist[-1]
            w_occ = self._warp(p_occ, p_pose, pose, True)
            both = seen & self._warp(p_seen, p_pose, pose, False)
            iou = aligned_iou(occ, w_occ, both)
            leak = leak_score(occ, w_occ, both, self.band_px)
            bad = ((self.gate in ("iou", "both") and iou < self.outlier_iou)
                   or (self.gate in ("leak", "both") and leak > self.leak_frac))
            if not bad:
                self.hist.append((t, occ, seen, pose))
                self.rejected = 0
                gate = "accept"
            else:
                self.rejected += 1
                gate = "reject"
                if self.patience is not None and self.rejected >= self.patience:
                    self.hist = [(t, occ, seen, pose)]
                    self.rejected = 0
                    gate = "reset"
        self.hist = self.hist[-self.window:]
        votes = np.zeros(occ.shape, dtype=np.int32)
        voters = np.zeros(occ.shape, dtype=np.int32)
        for _, h_occ, h_seen, h_pose in self.hist:
            if h_occ is occ:
                w_occ, w_seen = occ, seen
            else:
                w_occ = self._warp(h_occ, h_pose, pose, True)
                w_seen = self._warp(h_seen, h_pose, pose, False)
            votes += w_occ & w_seen
            voters += w_seen
        # obstacle where at least half of the frames that see the cell say
        # obstacle (ties to obstacle); unseen by every voter or out of the
        # current view -> obstacle
        out = (2 * votes >= voters) | (voters == 0) | ~seen
        return out, FilterStep(gate, iou, leak, [h[0] for h in self.hist])
