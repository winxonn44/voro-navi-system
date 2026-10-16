"""JFA in PyTorch, for the GPU (Apple MPS or CUDA) and as a CPU comparison.

Same algorithm and tie-break as the NumPy reference, bit-identical across
backends. Two paths:

- Eager (CPU, CUDA, and MPS as a fallback): vectorized int64 tensor ops, a
  line-by-line port of the NumPy reference. Each pass launches ~150 full-grid
  kernels, so it is bound by memory traffic, not arithmetic.
- Fused (MPS, the default there): one Metal kernel per pass, compiled with
  `torch.mps.compile_shader`, one thread per pixel, like the Numba loop. It
  works in int32 because Apple GPUs emulate 64-bit integer division: with
  int64 math the same fused kernel is ~10x slower (README, Q1). Instead of the
  packed int64 key it compares (d2, site) lexicographically, which orders
  candidates exactly the same way.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import BIG, NO_SITE


def _shifted(cur: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    h, w = cur.shape
    cand = torch.full_like(cur, NO_SITE)
    if abs(dy) >= h or abs(dx) >= w:
        return cand
    cand[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)] = cur[
        max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)
    ]
    return cand


def jfa_pass(cur: torch.Tensor, k: int, py: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    h, w = cur.shape
    hw = h * w
    big = torch.full_like(cur, BIG)
    best = big.clone()
    for dy in (-k, 0, k):
        for dx in (-k, 0, k):
            cand = _shifted(cur, dy, dx)
            sy = torch.div(cand, w, rounding_mode="floor")
            sx = cand - sy * w
            d2 = (py - sy) ** 2 + (px - sx) ** 2
            key = torch.where(cand >= 0, d2 * hw + cand, big)
            best = torch.minimum(best, key)
    return torch.where(best == BIG, torch.full_like(cur, NO_SITE), torch.remainder(best, hw))


_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// One JFA pass, one thread per pixel. `cur`/`out` hold flat site indices
// (int32, -1 = no site). Keeps the candidate with the smallest squared
// distance, then the smallest site index: the same order as the packed key
// d2 * H*W + site used by the other backends.
kernel void jfa_pass(device const int* cur [[buffer(0)]],
                     device int* out [[buffer(1)]],
                     constant long& k [[buffer(2)]],
                     constant long& h [[buffer(3)]],
                     constant long& w [[buffer(4)]],
                     uint idx [[thread_position_in_grid]]) {
    int K = int(k), H = int(h), W = int(w);
    int y = int(idx) / W;
    int x = int(idx) - y * W;
    uint best_d2 = 0xFFFFFFFFu;
    int best_s = -1;
    for (int a = -1; a <= 1; ++a) {
        int yy = y + a * K;
        if (yy < 0 || yy >= H) continue;
        for (int b = -1; b <= 1; ++b) {
            int xx = x + b * K;
            if (xx < 0 || xx >= W) continue;
            int s = cur[yy * W + xx];
            if (s < 0) continue;
            int sy = s / W;
            int sx = s - sy * W;
            int dy = y - sy, dx = x - sx;
            uint d2 = uint(dy * dy) + uint(dx * dx);
            if (d2 < best_d2 || (d2 == best_d2 && s < best_s)) {
                best_d2 = d2;
                best_s = s;
            }
        }
    }
    out[idx] = best_s;
}
"""

# int32 limits for the fused kernel: flat indices < 2^31, and dy*dy fits in
# int32 (|dy| <= 46340), so the uint sum of two squares cannot wrap.
_FUSED_MAX_PIXELS = 2**31 - 1
_FUSED_MAX_SIDE = 46340

_metal_lib = None


def _metal():
    global _metal_lib
    if _metal_lib is None:
        _metal_lib = torch.mps.compile_shader(_METAL_SRC)
    return _metal_lib


def fused_supported(height: int, width: int, device: str) -> bool:
    """Whether `run_steps` can use the fused Metal kernel for this grid."""
    h, w = height, width
    return (
        torch.device(device).type == "mps"
        and hasattr(torch.mps, "compile_shader")
        and 0 < h * w <= _FUSED_MAX_PIXELS
        and max(h, w) <= _FUSED_MAX_SIDE
    )


def _run_steps_fused(nearest: torch.Tensor, steps: list[int]) -> torch.Tensor:
    h, w = nearest.shape
    lib = _metal()
    cur = nearest.to(torch.int32, copy=True).contiguous()  # never write into the input
    nxt = torch.empty_like(cur)
    for k in steps:
        lib.jfa_pass(cur, nxt, k, h, w, threads=h * w)
        cur, nxt = nxt, cur
    return cur.to(torch.int64)


def to_device(nearest: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(nearest, dtype=np.int64)).to(device)


def run_steps(
    nearest: torch.Tensor, steps: list[int], fused: bool | None = None
) -> torch.Tensor:
    """Run the JFA passes on `nearest`'s device. `fused=None` picks the Metal
    kernel when it applies (MPS, grid within int32 limits); `fused=False`
    forces the eager int64 path; `fused=True` requires the kernel."""
    ok = fused_supported(*nearest.shape, str(nearest.device))
    if fused is None:
        fused = ok
    if fused:
        if not ok:
            raise ValueError("fused JFA needs an MPS tensor within int32 limits")
        return _run_steps_fused(nearest, steps)
    h, w = nearest.shape
    py, px = torch.meshgrid(
        torch.arange(h, dtype=torch.int64, device=nearest.device),
        torch.arange(w, dtype=torch.int64, device=nearest.device),
        indexing="ij",
    )
    cur = nearest
    for k in steps:
        cur = jfa_pass(cur, k, py, px)
    return cur


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs
