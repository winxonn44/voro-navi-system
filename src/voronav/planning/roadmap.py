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
along a minimum spanning tree of shortest safe paths between neighboring
pieces (`connect_pieces`; `bridge_lam` > 0 makes them the cheapest
clearance-weighted paths instead).
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from ..maps import boundary_sites
from .grid_search import reachable_components, source_field
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

    Pieces and regions are labeled by `reachable_components` with the exact
    move rule the searches use (regions: 4-connected; skeleton pieces:
    diagonal steps allowed where both corner cells are safe), so every
    piece of a region can actually be bridged and then traversed.
    In each region of `safe`: drop pieces smaller than `min_piece_px`
    (never the region's largest piece), then join the rest along a minimum
    spanning tree. One multi-source Dijkstra from every kept piece gives each
    safe pixel its nearest piece (`source_field`); two pieces whose
    territories touch are joined by the cheapest path through the two
    territories, and the tree keeps the cheapest set of such joins that
    connects the region (Mehlhorn's Steiner-tree approximation). That is one
    search for the whole map instead of one per bridge: the first version
    grew a tree from the largest piece one shortest path at a time, which
    took 25 s for the ~2,950 bridges of a MovingAI 8-room map.
    With bridge_lam > 0 (needs `clearance`), costs are step length times
    1 + bridge_lam / clearance, so bridges keep away from walls.
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
    pieces, n_pieces = reachable_components(skeleton, corner_ok=safe)
    if n_pieces <= 1:
        return out, stats
    regions, _ = reachable_components(safe)
    size = np.bincount(pieces.ravel(), minlength=n_pieces + 1)
    region_of = np.zeros(n_pieces + 1, dtype=np.int64)
    region_of[pieces[skeleton]] = regions[skeleton]
    # the largest piece of each region (ties: lowest label) is always kept
    order = np.lexsort((-size[1:], region_of[1:])) + 1  # stable: by region, largest first
    lead = order[np.r_[True, region_of[order[1:]] != region_of[order[:-1]]]]
    keep = size >= min_piece_px
    keep[0] = False
    keep[lead] = True
    dropped = skeleton & ~keep[pieces]
    out &= ~dropped
    stats["n_pruned"] = int(dropped.sum())
    n_kept = int(keep.sum())
    n_tree_regions = len(np.unique(region_of[keep]))
    if n_kept == n_tree_regions:
        return out, stats  # one piece per region already

    _, parent, owner, (cu, cv, cost) = source_field(safe, np.where(keep[pieces], pieces, 0), mult)
    flat_owner = owner.ravel()
    a, b = flat_owner[cu], flat_owner[cv]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    # cheapest contact per pair of pieces, then the spanning tree over them
    pick = np.lexsort((cost, hi, lo))
    pair = lo[pick] * (n_pieces + 1) + hi[pick]
    best = pick[np.r_[True, pair[1:] != pair[:-1]]]
    tree = minimum_spanning_tree(coo_matrix(
        (cost[best], (lo[best], hi[best])), shape=(n_pieces + 1, n_pieces + 1))).tocoo()
    if tree.nnz != n_kept - n_tree_regions:  # cannot happen inside one region; fail loudly
        raise RuntimeError("skeleton piece unreachable inside its own region")
    contact = {(int(x), int(y)): k for x, y, k in zip(lo[best], hi[best], best)}
    w = skeleton.shape[1]
    for x, y in sorted(zip(tree.row.tolist(), tree.col.tolist())):
        k = contact[(min(x, y), max(x, y))]
        chain = []
        for node in (int(cu[k]), int(cv[k])):  # each end back to its own piece
            while node >= 0:
                chain.append(node)
                node = int(parent[node])
        ys, xs = np.divmod(np.array(chain, dtype=np.int64), w)
        stats["n_bridges"] += 1
        new = ~out[ys, xs]
        stats["bridge_px"] += int(new.sum())  # pixels are unique within one bridge
        stats["bridges"][ys[new], xs[new]] = True
        out[ys, xs] = True
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
