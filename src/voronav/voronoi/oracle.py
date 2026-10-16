"""Brute-force exact discrete Voronoi diagram: the ground truth for JFA.

O(pixels x sites), chunked to bound memory. Uses the same packed-key
tie-break as the JFA backends, but compare JFA to it by *distance*: on tied
pixels JFA may never have seen the smallest-index site and keeps another
equidistant one, which is still a correct assignment.
"""

from __future__ import annotations

import numpy as np

from .common import NO_SITE

# int64 elements per chunk (pixels x sites); ~256 MB of scratch
_CHUNK_ELEMS = 32_000_000


def brute_force(sites: np.ndarray) -> np.ndarray:
    if sites.ndim != 2 or sites.dtype != bool:
        raise ValueError("sites must be a 2D bool array")
    h, w = sites.shape
    hw = h * w
    site_idx = np.flatnonzero(sites).astype(np.int64)
    if site_idx.size == 0:
        return np.full((h, w), NO_SITE, dtype=np.int64)
    sy, sx = np.divmod(site_idx, w)

    out = np.empty(hw, dtype=np.int64)
    chunk = max(1, _CHUNK_ELEMS // site_idx.size)
    for start in range(0, hw, chunk):
        pix = np.arange(start, min(start + chunk, hw), dtype=np.int64)
        py, px = np.divmod(pix, w)
        d2 = (py[:, None] - sy[None, :]) ** 2 + (px[:, None] - sx[None, :]) ** 2
        key = d2 * hw + site_idx[None, :]
        out[start : start + pix.size] = key.min(axis=1) % hw
    return out.reshape(h, w)
