"""Multithreaded CPU JFA with Numba: one pixel per loop iteration.

Two ways to hand rows to threads:
  dynamic (default): each thread repeatedly claims the next `chunk` rows from
    a shared atomic counter, so faster cores simply take more rows.
  static: `prange` over rows, which splits them into equal contiguous blocks.
On the M4 (4 performance + 6 efficiency cores) static splitting makes the
efficiency-core threads the stragglers, and 6 threads run slower than 4.
`numba.set_parallel_chunksize` does not fix this: it still gave each thread
exactly H / threads rows on both the workqueue and OpenMP layers (README, Q1).

Same packed-key tie-break as the NumPy reference, so the output is identical
for any thread count and either schedule.
"""

from __future__ import annotations

import os

import numba
import numpy as np
from numba import njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic

# Numba's default threading layer on macOS is OpenMP, and PyTorch ships its own
# libomp: loading both in one process aborts with "OMP: Error #15". The
# workqueue layer has no OpenMP dependency. Override with NUMBA_THREADING_LAYER.
if "NUMBA_THREADING_LAYER" not in os.environ:
    numba.config.THREADING_LAYER = "workqueue"

_BIG = np.iinfo(np.int64).max
SCHEDULES = ("dynamic", "static")
DEFAULT_CHUNK = 4  # rows per claim; 1, 4 and 16 measured within noise at 2048^2


@intrinsic
def _fetch_add(typingctx, arr, val):
    """Atomically `old = arr[0]; arr[0] += val; return old` (int64)."""
    sig = types.int64(arr, val)

    def codegen(context, builder, signature, args):
        aty = signature.args[0]
        ary = context.make_array(aty)(context, builder, args[0])
        zero = context.get_constant(types.intp, 0)
        ptr = cgutils.get_item_pointer(context, builder, aty, ary, [zero])
        return builder.atomic_rmw("add", ptr, args[1], "seq_cst")

    return sig, codegen


@njit(inline="always")
def _pixel(cur, y, x, k, h, w, hw):
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
    return -1 if best == _BIG else best % hw


@njit(parallel=True, cache=True)
def _pass(cur, out, k):
    h, w = cur.shape
    hw = h * w
    for y in prange(h):
        for x in range(w):
            out[y, x] = _pixel(cur, y, x, k, h, w, hw)


@njit(parallel=True, cache=True)
def _pass_dynamic(cur, out, k, counter, chunk, n_workers):
    h, w = cur.shape
    hw = h * w
    counter[0] = 0
    for _ in prange(n_workers):  # one iteration per thread; rows come from the counter
        while True:
            y0 = _fetch_add(counter, chunk)
            if y0 >= h:
                break
            for y in range(y0, min(y0 + chunk, h)):
                for x in range(w):
                    out[y, x] = _pixel(cur, y, x, k, h, w, hw)


def run_steps(
    nearest: np.ndarray,
    steps: list[int],
    threads: int | None = None,
    schedule: str = "dynamic",
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if threads is not None:
        numba.set_num_threads(threads)
    cur = np.ascontiguousarray(nearest, dtype=np.int64)
    nxt = np.empty_like(cur)
    counter = np.zeros(1, dtype=np.int64)
    n_workers = numba.get_num_threads()
    for k in steps:
        if schedule == "dynamic":
            _pass_dynamic(cur, nxt, k, counter, chunk, n_workers)
        else:
            _pass(cur, nxt, k)
        cur, nxt = nxt, cur
    return cur


def max_threads() -> int:
    return numba.config.NUMBA_NUM_THREADS
