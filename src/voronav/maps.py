"""Map sources: synthetic generators and loaders for real map formats.

Every map is a boolean occupancy grid `occ[H, W]` (True = obstacle) or, for
pure Voronoi tests, a boolean site grid (True = site). Sites for the
generalized Voronoi diagram of a map are its obstacle boundary pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import MapConfig

# MovingAI .map terrain characters (movingai.com/benchmarks/formats.html).
# '.', 'G' are passable ground; 'S' (swamp) is passable; '@', 'O', 'T'
# (trees) and 'W' (water) block a ground robot.
_MOVINGAI_FREE = set(".GS")
_MOVINGAI_BLOCKED = set("@OTW")


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def random_sites(height: int, width: int, n: int, seed: int = 0) -> np.ndarray:
    """n distinct uniformly random site pixels."""
    if n > height * width:
        raise ValueError("more sites than pixels")
    rng = np.random.default_rng(seed)
    flat = rng.choice(height * width, size=n, replace=False)
    sites = np.zeros(height * width, dtype=bool)
    sites[flat] = True
    return sites.reshape(height, width)


def random_obstacles(
    height: int, width: int, n_shapes: int, seed: int = 0, border: bool = True
) -> np.ndarray:
    """Occupancy grid of random axis-aligned rectangles and discs.

    The border wall keeps the free space bounded, so the GVD has no edges
    that run off the map.
    """
    rng = np.random.default_rng(seed)
    occ = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[0:height, 0:width]
    max_extent = max(4, min(height, width) // 8)
    for _ in range(n_shapes):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        if rng.random() < 0.5:
            hh, hw = rng.integers(2, max_extent, size=2)
            occ[max(0, cy - hh) : cy + hh, max(0, cx - hw) : cx + hw] = True
        else:
            r = rng.integers(2, max_extent)
            occ |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    if border:
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


def corridor(height: int, width: int, gap: int) -> np.ndarray:
    """Straight horizontal corridor of free width `gap` centered vertically.

    The exact maximum-clearance path is known (the centerline), which makes
    this the end-to-end fixture for the roadmap work in week 3.
    """
    if gap < 1 or gap > height - 2:
        raise ValueError("gap must fit inside the map with walls on both sides")
    occ = np.ones((height, width), dtype=bool)
    top = (height - gap) // 2
    occ[top : top + gap, :] = False
    return occ


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_movingai(path: str | Path) -> np.ndarray:
    """Parse a MovingAI .map file into an occupancy grid."""
    lines = Path(path).read_text().splitlines()
    header: dict[str, str] = {}
    i = 0
    while i < len(lines) and lines[i].strip() != "map":
        parts = lines[i].split()
        if len(parts) == 2:
            header[parts[0]] = parts[1]
        i += 1
    if i == len(lines):
        raise ValueError(f"{path}: missing 'map' line")
    try:
        height, width = int(header["height"]), int(header["width"])
    except KeyError as e:
        raise ValueError(f"{path}: missing header field {e}") from None

    rows = lines[i + 1 : i + 1 + height]
    if len(rows) != height:
        raise ValueError(f"{path}: expected {height} rows, got {len(rows)}")
    occ = np.zeros((height, width), dtype=bool)
    for y, row in enumerate(rows):
        if len(row) < width:
            raise ValueError(f"{path}: row {y} has {len(row)} < {width} chars")
        for x, ch in enumerate(row[:width]):
            if ch in _MOVINGAI_BLOCKED:
                occ[y, x] = True
            elif ch not in _MOVINGAI_FREE:
                raise ValueError(f"{path}: unknown terrain {ch!r} at ({y}, {x})")
    return occ


# HouseExpo resolution: 20 px/m = 5 cm cells. The dataset's own viewer uses
# 100 px/m, which puts a typical 10-25 m house at 1000-2500 px; at 20 px/m the
# same house is 200-500 px, inside the ~512 px budget of the Q2 benchmark,
# while a 0.8 m door is still 16 px wide and a 10 cm interior wall 2 px thick.
HOUSEEXPO_PX_PER_METER = 20.0


def load_houseexpo(
    path: str | Path, px_per_meter: float = HOUSEEXPO_PX_PER_METER, pad: int = 2
) -> np.ndarray:
    """Rasterize a HouseExpo floor plan (github.com/TeaganLi/HouseExpo) into
    an occupancy grid.

    The JSON's `verts` is one closed contour of the house in meters; the
    inside is free floor and everything outside (including walls, which the
    contour wraps around) is obstacle. A pixel is free when its center lies
    inside the contour (even-odd rule). Rows follow +y and columns +x, as in
    the dataset's own PNG renderer, with `pad` obstacle pixels on every side
    so the free space is always enclosed.
    """
    import json

    data = json.loads(Path(path).read_text())
    try:
        verts = np.asarray(data["verts"], dtype=np.float64)
    except KeyError:
        raise ValueError(f"{path}: missing 'verts'") from None
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError(f"{path}: 'verts' must be a list of >= 3 (x, y) points")
    if px_per_meter <= 0 or pad < 0:
        raise ValueError("px_per_meter must be > 0 and pad >= 0")

    lo = verts.min(axis=0)
    px = (verts - lo) * px_per_meter  # contour in pixel units, origin at bbox min
    width = int(np.ceil(px[:, 0].max())) + 2 * pad
    height = int(np.ceil(px[:, 1].max())) + 2 * pad

    # Scanline fill: for each row center, x-crossings of the edges that
    # straddle it (half-open in y, so a vertex on the scanline counts once).
    x0, y0 = px[:, 0], px[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    occ = np.ones((height, width), dtype=bool)
    xc = np.arange(width) - pad + 0.5
    for row in range(height):
        yc = row - pad + 0.5
        hit = (y0 <= yc) != (y1 <= yc)
        if not hit.any():
            continue
        t = (yc - y0[hit]) / (y1[hit] - y0[hit])
        xs = np.sort(x0[hit] + t * (x1[hit] - x0[hit]))
        # even-odd: a center is inside when an odd number of crossings lie left of it
        inside = np.searchsorted(xs, xc) % 2 == 1
        occ[row] = ~inside
    return occ


def load_image_mask(path: str | Path, free_is_bright: bool = True) -> np.ndarray:
    """Load a binary PNG/JPG mask (e.g. a SAM free-space mask, a HouseExpo
    render) as an occupancy grid, thresholding at mid-gray."""
    from PIL import Image

    gray = np.asarray(Image.open(path).convert("L"))
    bright = gray >= 128
    return ~bright if free_is_bright else bright


# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------


def boundary_sites(occ: np.ndarray) -> np.ndarray:
    """Obstacle pixels 4-adjacent to free space.

    Interior obstacle pixels never win a nearest-site query from free space
    (a boundary pixel is always at least as close), so dropping them gives the
    same distance field on free space with far fewer sites.
    """
    free = ~occ
    near_free = np.zeros_like(occ)
    near_free[1:, :] |= free[:-1, :]
    near_free[:-1, :] |= free[1:, :]
    near_free[:, 1:] |= free[:, :-1]
    near_free[:, :-1] |= free[:, 1:]
    return occ & near_free


def build_map(cfg: MapConfig) -> np.ndarray:
    """Site grid for `points`, occupancy grid for everything else."""
    if cfg.kind == "points":
        return random_sites(cfg.height, cfg.width, cfg.n_sites, cfg.seed)
    if cfg.kind == "obstacles":
        return random_obstacles(cfg.height, cfg.width, cfg.n_sites, cfg.seed)
    if cfg.kind == "corridor":
        return corridor(cfg.height, cfg.width, gap=max(1, cfg.height // 4))
    if cfg.kind == "movingai":
        return load_movingai(cfg.path)
    return load_image_mask(cfg.path)
