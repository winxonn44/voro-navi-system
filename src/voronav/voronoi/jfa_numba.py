"""Multithreaded CPU JFA with Numba: one pixel per loop iteration, rows split
across threads with `prange`.

Same packed-key tie-break as the NumPy reference, so the output is identical
for any thread count.
"""

from __future__ import annotations

import os

import numba
import numpy as np
from numba import njit, prange

# Numba's default threading layer on macOS is OpenMP, and PyTorch ships its own
# libomp: loading both in one process aborts with "OMP: Error #15". The
# workqueue layer has no OpenMP dependency. Override with NUMBA_THREADING_LAYER.
if "NUMBA_THREADING_LAYER" not in os.environ:
    numba.config.THREADING_LAYER = "workqueue"

_BIG = np.iinfo(np.int64).max


@njit(parallel=True, cache=True)
def _pass(cur, out, k):
    h, w = cur.shape
    hw = h * w
    for y in prange(h):
        for x in range(w):
            best = _BIG
            for a in range(3):
                yy = y + (a - 1) * k
                if yy < 0 or yy >= h:
                    continue
                for b in range(3):
                    xx = x + (b - 1) * k
                    if xx < 0 or xx >= w:
                        continue
                    s = cur[yy, xx]
                    if s < 0:
                        continue
                    sy = s // w
                    sx = s - sy * w
                    d2 = (y - sy) * (y - sy) + (x - sx) * (x - sx)
                    key = d2 * hw + s
                    if key < best:
                        best = key
            out[y, x] = -1 if best == _BIG else best % hw


def run_steps(nearest: np.ndarray, steps: list[int], threads: int | None = None) -> np.ndarray:
    if threads is not None:
        numba.set_num_threads(threads)
    cur = np.ascontiguousarray(nearest, dtype=np.int64)
    nxt = np.empty_like(cur)
    for k in steps:
        _pass(cur, nxt, k)
        cur, nxt = nxt, cur
    return cur


def max_threads() -> int:
    return numba.config.NUMBA_NUM_THREADS
