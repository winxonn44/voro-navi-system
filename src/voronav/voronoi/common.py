"""Shared JFA conventions, so every backend produces bit-identical output.

Representation: `nearest[H, W]` (int64) holds the flat index `y * W + x` of
the nearest site, or -1 where no site has been propagated yet.

Tie-break: among equidistant candidates, the site with the smallest flat
index wins. Every backend compares the packed key `d2 * (H * W) + site`,
which orders by squared distance first and site index second, using integer
arithmetic only. That makes the result independent of candidate visit order
and thread scheduling, so backends can be checked against each other with
`array_equal` rather than a tolerance. (It does not make JFA agree with the
exact oracle on ties: a pixel only chooses among the candidates JFA
propagated to it.)
"""

from __future__ import annotations

import numpy as np

from ..config import VARIANTS

NO_SITE = -1
BIG = np.iinfo(np.int64).max


def init_nearest(sites: np.ndarray) -> np.ndarray:
    """Seed grid: each site points at itself, everything else is NO_SITE."""
    if sites.ndim != 2 or sites.dtype != bool:
        raise ValueError("sites must be a 2D bool array")
    h, w = sites.shape
    flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
    return np.where(sites, flat, NO_SITE)


def step_schedule(height: int, width: int, variant: str = "jfa") -> list[int]:
    """JFA step lengths.

    Standard JFA (Rong & Tan 2006) uses k = N/2, N/4, ..., 1 with N the grid
    side rounded up to a power of two. The variants add cheap passes that
    repair most of the residual errors:
      jfa+1: one extra k=1 pass at the end
      jfa+2: extra k=2 and k=1 passes at the end
      1+jfa: one k=1 pass before the standard schedule
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    n = 1
    while n < max(height, width):
        n *= 2
    steps = []
    k = n // 2
    while k >= 1:
        steps.append(k)
        k //= 2
    if variant == "jfa+1":
        steps += [1]
    elif variant == "jfa+2":
        steps += [2, 1]
    elif variant == "1+jfa":
        steps = [1] + steps
    return steps


def distance_sq(nearest: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from each pixel to its assigned site
    (int64, -1 where unassigned)."""
    h, w = nearest.shape
    py, px = np.indices((h, w), dtype=np.int64)
    sy, sx = np.divmod(nearest, w)
    d2 = (py - sy) ** 2 + (px - sx) ** 2
    return np.where(nearest >= 0, d2, -1)
