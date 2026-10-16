"""JFA in PyTorch, for the GPU (Apple MPS or CUDA) and as a CPU comparison.

Same algorithm and tie-break as the NumPy reference; everything stays int64
on the device so results are exact and bit-identical across backends.
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


def to_device(nearest: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(nearest, dtype=np.int64)).to(device)


def run_steps(nearest: torch.Tensor, steps: list[int]) -> torch.Tensor:
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
