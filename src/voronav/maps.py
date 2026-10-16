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
