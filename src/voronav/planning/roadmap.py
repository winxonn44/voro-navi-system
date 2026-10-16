"""Generalized Voronoi diagram (GVD) roadmap from an occupancy grid.

Pipeline: obstacle boundary pixels are the Voronoi sites -> JFA gives every
pixel its nearest site and clearance -> a pixel lies on the GVD where its
nearest site and a neighbor's nearest site belong to "different sides" of
free space.

"Different sides" has two cases:
  - The two nearest sites belong to different obstacles (8-connected
    components of the occupancy grid): always a GVD edge. This is the
    textbook GVD, equidistant to two distinct obstacles.
  - Both sites belong to the same obstacle (a concave pocket, or anything
    attached to the border wall): an edge only if the sites subtend a large
    angle at the pixel (the object-angle criterion of Attali & Montanvert /
    Shaked & Bruckstein for pruning discrete medial axes), plus a raster
    tolerance so pixel staircases on curved walls do not sprout fringe.

The angle test alone was the first version. It broke real GVD edges where
two distinct obstacles lie in similar directions from the pixel (they subtend
under 60 degrees but are still different obstacles), which split one free
region's roadmap into disconnected pieces. Labeling alone fails the other
way: every obstacle touching the border wall merges into one component, so
obstacles are labeled with the outer 1-px frame removed (see `obstacle_ids`).

Finally, the angle test can still cut a real connection: through a door in
a wall that is one obstacle, the door's skeleton line fades out a few pixels
into each room as the two door jambs subtend a shrinking angle. So the
roadmap is *repaired*, not pruned: within each free region, pieces are joined
to the largest one by the shortest safe path to the nearest piece (`connect_pieces`;
`bridge_lam` > 0 makes it the cheapest clearance-weighted path instead).
Only fragments under `min_piece_px` are dropped. `Roadmap.n_bridges`,
`bridge_px` and `n_pruned` record what the repair did, so its effect on path
clearance can be measured rather than hidden. (The first version kept only
the largest piece per region; a two-room test caught it deleting the door.)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, maximum_filter

from ..maps import boundary_sites
from .grid_search import search
from ..voronoi import distance_sq, voronoi

# Squared site separation must exceed this as well: adjacent and diagonal
# wall pixels (sep^2 = 1 or 2) never form a GVD edge, while a one-pixel gap
# between two obstacles (sites 2 px apart, sep^2 = 4) still does.
_MIN_SEP2 = 3
_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Roadmap:
    occ: np.ndarray        # bool [H, W], True = obstacle
    nearest: np.ndarray    # int64 [H, W], nearest boundary site (flat), -1 if none
    clearance: np.ndarray  # float64 [H, W], distance to nearest obstacle (px); inf if none
    safe: np.ndarray       # bool [H, W], free and clearance >= robot_radius
    skeleton: np.ndarray   # bool [H, W], GVD pixels inside `safe`
    robot_radius: float
    build_s: float         # wall-clock time to build (JFA + extraction)
    n_pruned: int = 0      # skeleton pixels dropped as tiny fragments
    n_bridges: int = 0     # paths added to join skeleton pieces in one free region
    bridge_px: int = 0     # pixels those bridges added
    bridges: np.ndarray | None = None  # bool [H, W], the pixels bridges added

    @property
    def n_nodes(self) -> int:
        return int(self.skeleton.sum())


def obstacle_ids(occ: np.ndarray) -> np.ndarray:
    """Obstacle component labels (0 = free), ignoring the outer 1-px frame
    as a connector.

    A map's border wall would otherwise merge every obstacle that touches it
    into a single component. Frame pixels take the label of an obstacle they
    touch on the inside, or else one label per side of the map.
    """
    h, w = occ.shape
    if h < 3 or w < 3:
        ids, _ = label(occ, structure=_EIGHT)
        return ids
    inner = occ.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    ids, n = label(inner, structure=_EIGHT)
    frame = np.zeros_like(occ)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    frame &= occ
    side = np.zeros_like(ids)
    side[0, :], side[-1, :] = n + 1, n + 2
    side[1:-1, 0], side[1:-1, -1] = n + 3, n + 4
    touching = maximum_filter(ids, size=3)
    return np.where(frame, np.where(touching > 0, touching, side), ids)


def connect_pieces(
    skeleton: np.ndarray,
    safe: np.ndarray,
    min_piece_px: int = 8,
    clearance: np.ndarray | None = None,
    bridge_lam: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Make the skeleton one connected piece per free region.

    In each 8-connected region of `safe`: drop pieces smaller than
    `min_piece_px` (unless it is the region's only piece), then grow a tree
    from the largest piece, repeatedly adding the shortest safe path from the
    tree to the nearest remaining piece (multi-source Dijkstra). With
    bridge_lam > 0 (needs `clearance`), bridges are the cheapest paths under
    step cost 1 + bridge_lam / clearance instead of the shortest ones.
    Returns the repaired skeleton and {"n_pruned", "n_bridges", "bridge_px",
    "bridges"} (the last is the mask of pixels the bridges added).
    """
    if bridge_lam < 0:
        raise ValueError("bridge_lam must be >= 0")
    if bridge_lam > 0 and clearance is None:
        raise ValueError("bridge_lam > 0 needs the clearance field")
    mult = None
    if bridge_lam > 0:
        with np.errstate(divide="ignore"):
            mult = np.where(clearance > 0, 1.0 + bridge_lam / clearance, np.inf)
    out = skeleton.copy()
    stats = {"n_pruned": 0, "n_bridges": 0, "bridge_px": 0,
             "bridges": np.zeros_like(skeleton, dtype=bool)}
    pieces, n_pieces = label(skeleton, structure=_EIGHT)
    if n_pieces <= 1:
        return out, stats
    regions, _ = label(safe, structure=_EIGHT)
    size = np.bincount(pieces.ravel(), minlength=n_pieces + 1)
    region_of = np.zeros(n_pieces + 1, dtype=np.int64)
    region_of[pieces[skeleton]] = regions[skeleton]
    by_region: dict[int, list[int]] = {}
    for piece in range(1, n_pieces + 1):
        by_region.setdefault(int(region_of[piece]), []).append(piece)

    for r, members in by_region.items():
        members.sort(key=lambda p: -size[p])
        largest, rest = members[0], members[1:]
        for p in rest:
            if size[p] < min_piece_px:
                out[pieces == p] = False
                stats["n_pruned"] += int(size[p])
        remaining = [p for p in rest if size[p] >= min_piece_px]
        if not remaining:
            continue
        region = regions == r
        tree = pieces == largest
        todo = np.isin(pieces, remaining)
        while todo.any():
            res = search(safe & region, start_mask=tree, goal_mask=todo, cost_mult=mult)
            if res.path is None:  # cannot happen inside one region; fail loudly
                raise RuntimeError("skeleton piece unreachable inside its own region")
            ys, xs = res.path[:, 0], res.path[:, 1]
            stats["n_bridges"] += 1
            stats["bridge_px"] += int((~out[ys, xs]).sum())
            stats["bridges"][ys, xs] |= ~out[ys, xs]
            out[ys, xs] = True
            reached = pieces[res.path[-1, 0], res.path[-1, 1]]
            joined = (pieces == reached)
            tree |= joined
            tree[ys, xs] = True
            todo &= ~joined
    return out, stats


def gvd_mask(
    occ: np.ndarray,
    nearest: np.ndarray,
    d2: np.ndarray,
    min_angle_deg: float = 60.0,
    raster_slack_px: float = 2.0,
) -> np.ndarray:
    """Pixels on the generalized Voronoi diagram of the free space.

    A 4-adjacent pair of free pixels (p, q) with nearest sites s_p, s_q is a
    GVD edge when s_p and s_q lie on different obstacles, or on the same
    obstacle with |s_p - s_q| > 2 sin(theta / 2) * d + slack, where d is the
    larger of the two clearances (the sites subtend at least `min_angle_deg`).
    Both pixels of the pair are marked, which keeps the skeleton 8-connected
    (marking one side can open diagonal gaps); it is up to 2 px thick.
    """
    if not 0.0 < min_angle_deg <= 180.0:
        raise ValueError("min_angle_deg must be in (0, 180]")
    h, w = occ.shape
    free = ~occ & (nearest >= 0)
    obstacle_id = obstacle_ids(occ)
    site_obstacle = np.where(nearest >= 0, obstacle_id.reshape(-1)[np.maximum(nearest, 0)], 0)
    sy, sx = np.divmod(nearest, w)
    rho = 2.0 * math.sin(math.radians(min_angle_deg) / 2.0)
    skel = np.zeros((h, w), dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        p = (slice(0, h - dy), slice(0, w - dx))
        q = (slice(dy, h), slice(dx, w))
        sep2 = (sy[p] - sy[q]) ** 2 + (sx[p] - sx[q]) ** 2
        dmax = np.sqrt(np.maximum(np.maximum(d2[p], d2[q]), 0))
        different = site_obstacle[p] != site_obstacle[q]
        wide = sep2 > (rho * dmax + raster_slack_px) ** 2
        edge = free[p] & free[q] & (sep2 > _MIN_SEP2) & (different | wide)
        skel[p] |= edge
        skel[q] |= edge
    return skel


def build_roadmap(
    occ: np.ndarray,
    robot_radius: float = 1.0,
    min_angle_deg: float = 60.0,
    raster_slack_px: float = 2.0,
    min_piece_px: int = 8,
    bridge_lam: float = 0.0,
    backend: str = "numba",
    variant: str = "jfa+2",
    device: str = "cpu",
) -> Roadmap:
    """Build the GVD roadmap. `jfa+2` is the default variant because it was
    exact on every Q1 instance; a wrong nearest site can break or bend the
    skeleton, which matters more here than a few microseconds."""
    if occ.ndim != 2 or occ.dtype != bool:
        raise ValueError("occ must be a 2D bool array")
    t0 = time.perf_counter()
    nearest = voronoi(boundary_sites(occ), backend=backend, variant=variant, device=device)
    d2 = distance_sq(nearest)
    clearance = np.where(d2 >= 0, np.sqrt(np.maximum(d2, 0)), np.inf)
    safe = ~occ & (clearance >= robot_radius)
    skeleton = gvd_mask(occ, nearest, d2, min_angle_deg, raster_slack_px) & safe
    skeleton, repair = connect_pieces(skeleton, safe, min_piece_px, clearance, bridge_lam)
    return Roadmap(
        occ=occ,
        nearest=nearest,
        clearance=clearance,
        safe=safe,
        skeleton=skeleton,
        robot_radius=robot_radius,
        build_s=time.perf_counter() - t0,
        **repair,
    )
