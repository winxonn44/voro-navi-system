"""Segmentation-error models for Q3: corrupt a true map, plan on the result.

The perception pipeline segments *floor* (free space), so every model is a
mask error on the floor mask, applied to an occupancy grid (True = obstacle).
Two directions matter and they are not symmetric:

- false floor (obstacle marked free) is the dangerous error: the planner may
  route through something real;
- false obstacle (floor marked blocked) fails safe, but can disconnect the
  map and strand the robot.

Every model has the signature `fn(occ, level, rng) -> new_occ`: it returns a
new bool grid of the same shape, never mutates `occ`, and `level = 0` returns
an identical copy. Optional keyword arguments tune secondary knobs.

The outermost ring of pixels is treated as the known extent of the view, not
as a segmentation output: obstacle pixels on the map edge are never freed.
That keeps maps with a border frame bounded no matter the corruption.

See docs/Q3_ERRORS.md for what each model stands for in SAM terms.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label

from .voronoi import voronoi

__all__ = [
    "CORRUPTIONS",
    "boundary_noise",
    "boundary_shift",
    "mask_error_stats",
    "region_dropout",
    "signed_floor_distance",
    "thin_obstacle_miss",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(occ: np.ndarray) -> None:
    if occ.ndim != 2 or occ.dtype != bool:
        raise ValueError("occ must be a 2D bool array")


def _keep_frame(occ: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Re-block any edge pixel that was an obstacle in `occ`."""
    new[0, :] |= occ[0, :]
    new[-1, :] |= occ[-1, :]
    new[:, 0] |= occ[:, 0]
    new[:, -1] |= occ[:, -1]
    return new


def signed_floor_distance(occ: np.ndarray) -> np.ndarray:
    """Signed distance (px) to the floor/obstacle boundary, positive on floor.

    The boundary sits halfway between pixel centers, so pixels touching it
    are at +-0.5 and the pixel `d` steps into a wall is at -(d - 0.5).
    Infinite when the map is all floor or all obstacle.
    """
    if occ.all():
        return np.full(occ.shape, -np.inf)
    if not occ.any():
        return np.full(occ.shape, np.inf)
    return np.where(
        occ, -(distance_transform_edt(occ) - 0.5), distance_transform_edt(~occ) - 0.5
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def boundary_shift(occ: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Floor mask grows into walls (level > 0) or shrinks from them (< 0).

    Euclidean: with level = r > 0, an obstacle pixel turns into floor when
    the nearest floor pixel center is within r; with level = -r, a floor
    pixel turns into obstacle when the nearest obstacle pixel center is
    within r (a disc dilation of the obstacles). Deterministic; `rng` is
    unused and only there for the common signature.
    """
    _check(occ)
    if level == 0 or occ.all() or not occ.any():
        return occ.copy()
    if level > 0:
        new = occ & (distance_transform_edt(occ) > level)
    else:
        new = occ | (distance_transform_edt(~occ) <= -level)
    return _keep_frame(occ, new)


def region_dropout(
    occ: np.ndarray,
    level: float,
    rng: np.random.Generator,
    cell_px: float | None = None,
) -> np.ndarray:
    """Remove blob-shaped chunks of floor totalling ~`level` of the free area.

    Free space is partitioned into Voronoi cells around random floor seeds
    (JFA, numpy backend), then whole cells are marked obstacle in random
    order, each one kept only if it brings the removed total closer to the
    target. `cell_px` is the typical cell side; by default it scales with
    the map (sqrt(free pixels) / 6, so roughly 36 cells), which keeps the
    error from the target to a fraction of one cell.
    """
    _check(occ)
    level = float(np.clip(level, 0.0, 1.0))
    free = ~occ
    n_free = int(free.sum())
    if level == 0 or n_free == 0:
        return occ.copy()
    if cell_px is None:
        cell_px = max(4.0, np.sqrt(n_free) / 6.0)
    n_cells = int(np.clip(round(n_free / cell_px**2), 1, n_free))

    seeds = rng.choice(np.flatnonzero(free), size=n_cells, replace=False)
    sites = np.zeros(occ.size, dtype=bool)
    sites[seeds] = True
    nearest = voronoi(sites.reshape(occ.shape))
    cell_of_free = nearest[free]
    cells, sizes = np.unique(cell_of_free, return_counts=True)

    target = level * n_free
    removed = 0
    drop = []
    for i in rng.permutation(len(cells)):
        if removed + sizes[i] / 2 < target:
            drop.append(cells[i])
            removed += sizes[i]
    new = occ.copy()
    new[free] = np.isin(cell_of_free, drop)
    return new


def thin_obstacle_miss(occ: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Obstacle components thinner than `level` px vanish into the floor.

    Width of a component = 2 x its largest distance-transform value (the
    diameter of its biggest inscribed disc, measured center to center), so
    a w-px bar has width w for even w and w + 1 for odd w. Components are
    8-connected, and a component touching the map edge (the border frame)
    is always kept. Deterministic; `rng` is unused.
    """
    _check(occ)
    if level <= 0 or occ.all() or not occ.any():
        return occ.copy()
    comp, n = label(occ, structure=np.ones((3, 3), dtype=bool))
    depth = distance_transform_edt(occ)
    max_depth = np.zeros(n + 1)
    np.maximum.at(max_depth, comp.ravel(), depth.ravel())
    erase = 2.0 * max_depth < level
    erase[0] = False
    edge = np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]])
    erase[edge] = False
    new = occ & ~erase[comp]
    return _keep_frame(occ, new)


def boundary_noise(
    occ: np.ndarray,
    level: float,
    rng: np.random.Generator,
    corr_px: float | None = None,
) -> np.ndarray:
    """Jagged, spatially correlated boundary errors of amplitude ~`level` px.

    A white-noise field is smoothed with a Gaussian of width `corr_px`
    (default max(2, 2 x level)), standardised and clipped to [-1, 1], then
    a pixel is floor iff signed_floor_distance + level x field > 0. Errors
    go both ways and only pixels within `level` of the true boundary can
    flip, so the error stays inside a band of half-width `level`.
    """
    _check(occ)
    if level <= 0 or occ.all() or not occ.any():
        return occ.copy()
    if corr_px is None:
        corr_px = max(2.0, 2.0 * level)
    field = gaussian_filter(rng.standard_normal(occ.shape), corr_px, mode="wrap")
    field = np.clip(field / (field.std() + 1e-12), -1.0, 1.0)
    new = signed_floor_distance(occ) + level * field <= 0
    return _keep_frame(occ, new)


CORRUPTIONS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "boundary_shift": boundary_shift,
    "region_dropout": region_dropout,
    "thin_obstacle_miss": thin_obstacle_miss,
    "boundary_noise": boundary_noise,
}


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------


def _boundary(occ: np.ndarray) -> np.ndarray:
    """Obstacle pixels with a 4-neighbour of floor (the map edge doesn't count)."""
    return occ & ~binary_erosion(occ, border_value=1)


def mask_error_stats(true_occ: np.ndarray, est_occ: np.ndarray) -> dict:
    """How far an estimated occupancy grid is from the truth, as a floor mask.

    free_iou            IoU of the floor masks (1.0 if both have no floor).
    false_free_frac     true obstacle pixels marked floor / true obstacle
                        pixels. The dangerous error.
    false_obstacle_frac true floor pixels marked obstacle / true floor pixels.
    boundary_error_px   mean distance from each estimated boundary pixel to
                        the nearest true boundary pixel (0 if both have no
                        boundary, nan if only one does).
    boundary_error_max_px  the same, max instead of mean.
    n_false_free, n_false_obstacle  raw pixel counts.
    """
    if true_occ.shape != est_occ.shape:
        raise ValueError("shape mismatch")
    t_free, e_free = ~true_occ, ~est_occ
    union = int((t_free | e_free).sum())
    n_ff = int((true_occ & e_free).sum())
    n_fo = int((t_free & est_occ).sum())
    n_obs, n_flr = int(true_occ.sum()), int(t_free.sum())

    tb, eb = _boundary(true_occ), _boundary(est_occ)
    if not tb.any() and not eb.any():
        b_mean = b_max = 0.0
    elif not tb.any() or not eb.any():
        b_mean = b_max = float("nan")
    else:
        d = distance_transform_edt(~tb)[eb]
        b_mean, b_max = float(d.mean()), float(d.max())

    return {
        "free_iou": int((t_free & e_free).sum()) / union if union else 1.0,
        "false_free_frac": n_ff / n_obs if n_obs else 0.0,
        "false_obstacle_frac": n_fo / n_flr if n_flr else 0.0,
        "boundary_error_px": b_mean,
        "boundary_error_max_px": b_max,
        "n_false_free": n_ff,
        "n_false_obstacle": n_fo,
    }
