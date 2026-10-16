"""Reference JFA in vectorized NumPy.

This is the readable version; the Numba and Torch backends must match it
bit for bit. Each pass is Jacobi-style: all reads come from the previous
pass's grid, never from pixels already updated in the current pass.
"""

from __future__ import annotations

import numpy as np

from .common import BIG, NO_SITE


def _shifted(cur: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """cand[y, x] = cur[y + dy, x + dx], NO_SITE where out of bounds."""
    h, w = cur.shape
    cand = np.full_like(cur, NO_SITE)
    if abs(dy) >= h or abs(dx) >= w:
        return cand
    dst_y = slice(max(0, -dy), h - max(0, dy))
    src_y = slice(max(0, dy), h + min(0, dy))
    dst_x = slice(max(0, -dx), w - max(0, dx))
    src_x = slice(max(0, dx), w + min(0, dx))
    cand[dst_y, dst_x] = cur[src_y, src_x]
    return cand


def jfa_pass(cur: np.ndarray, k: int, py: np.ndarray, px: np.ndarray) -> np.ndarray:
    h, w = cur.shape
    hw = h * w
    best = np.full_like(cur, BIG)
    for dy in (-k, 0, k):
        for dx in (-k, 0, k):
            cand = _shifted(cur, dy, dx)
            sy, sx = np.divmod(cand, w)
            d2 = (py - sy) ** 2 + (px - sx) ** 2
            key = np.where(cand >= 0, d2 * hw + cand, BIG)
            np.minimum(best, key, out=best)
    return np.where(best == BIG, NO_SITE, best % hw)


def run_steps(nearest: np.ndarray, steps: list[int]) -> np.ndarray:
    h, w = nearest.shape
    py, px = np.indices((h, w), dtype=np.int64)
    cur = nearest
    for k in steps:
        cur = jfa_pass(cur, k, py, px)
    return cur
