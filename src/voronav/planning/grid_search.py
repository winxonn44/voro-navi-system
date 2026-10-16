"""8-connected grid A* / Dijkstra, compiled with Numba.

One search routine serves every grid planner here:
  - grid A*: passable = safe free space, single goal, octile heuristic
  - roadmap search: passable = GVD skeleton pixels, single goal
  - roadmap connection: passable = safe free space, goal = *any* skeleton
    pixel, no heuristic (multi-target Dijkstra)

Moves cost 1 (orthogonal) or sqrt(2) (diagonal). A diagonal move may not cut
a corner: both orthogonal neighbors it passes must be in `corner_ok` (the
MovingAI benchmark rule). `corner_ok` is separate from `passable` so the
roadmap search can step diagonally along a skeleton line while still being
checked against the real free space.

Optional per-pixel step-cost multiplier (`cost_mult`, all >= 1): a move
costs its length times the multiplier of the pixel it enters, e.g.
1 + lambda / clearance for clearance-weighted A*. Because every multiplier is
>= 1, the octile heuristic still never overestimates, so A* stays optimal.
Without `cost_mult` the search takes the original branch, so unweighted
results are bit-identical to the version before weights existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# When passable == corner_ok (plain grid search through free space), a
# diagonal move needs both corner cells passable, so reachability is exactly
# 4-connectivity. Never label search regions with a 3x3 block of ones
# (8-connectivity): a "region" can then span a diagonal pinch no path can
# cross (MovingAI dao/lak303d: 4 regions 8-connected, 24 reachable).
REACHABILITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def reachable_components(mask: np.ndarray, corner_ok: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Label pixels of `mask` by what `search(mask, corner_ok=...)` can reach:
    orthogonal neighbors always connect; diagonal neighbors connect only when
    both corner cells are in `corner_ok` (default: `mask` itself, which is
    4-connectivity). Returns (labels with 0 = not in mask, count), like
    scipy.ndimage.label. Needed for the roadmap, whose diagonal steps are
    checked against free space rather than against the skeleton."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mask = np.asarray(mask, dtype=bool)
    corner_ok = mask if corner_ok is None else np.asarray(corner_ok, dtype=bool)
    h, w = mask.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols = [], []

    def link(a_sl, b_sl, ok):
        rows.append(idx[a_sl][ok])
        cols.append(idx[b_sl][ok])

    # right, down
    link((slice(None), slice(0, w - 1)), (slice(None), slice(1, w)), mask[:, :-1] & mask[:, 1:])
    link((slice(0, h - 1), slice(None)), (slice(1, h), slice(None)), mask[:-1, :] & mask[1:, :])
    # down-right: corners (y, x+1) and (y+1, x); down-left: (y, x-1) and (y+1, x)
    dr = mask[:-1, :-1] & mask[1:, 1:] & corner_ok[:-1, 1:] & corner_ok[1:, :-1]
    link((slice(0, h - 1), slice(0, w - 1)), (slice(1, h), slice(1, w)), dr)
    dl = mask[:-1, 1:] & mask[1:, :-1] & corner_ok[:-1, :-1] & corner_ok[1:, 1:]
    link((slice(0, h - 1), slice(1, w)), (slice(1, h), slice(0, w - 1)), dl)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(h * w, h * w))
    _, comp = connected_components(graph, directed=False)
    # relabel components of mask pixels as 1..n in scan order, 0 elsewhere
    comp = comp.reshape(h, w)
    uniq, first = np.unique(comp[mask], return_index=True)
    order = np.argsort(first)
    remap = np.zeros(comp.max() + 1, dtype=np.int64)
    remap[uniq[order]] = np.arange(1, len(uniq) + 1)
    return np.where(mask, remap[comp], 0), int(len(uniq))

_DY = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
_DX = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)


@njit(cache=True)
def _octile(y, x, gy, gx):
    dy = abs(y - gy)
    dx = abs(x - gx)
    return (dy + dx) + (np.sqrt(2.0) - 2.0) * min(dy, dx)


@njit(cache=True)
def _push(keys, vals, size, key, val):
    """Binary min-heap push; grows the arrays when full. Returns new arrays."""
    if size == keys.shape[0]:
        nk = np.empty(keys.shape[0] * 2, dtype=keys.dtype)
        nv = np.empty(vals.shape[0] * 2, dtype=vals.dtype)
        nk[:size] = keys[:size]
        nv[:size] = vals[:size]
        keys, vals = nk, nv
    i = size
    keys[i] = key
    vals[i] = val
    while i > 0:
        parent = (i - 1) // 2
        if keys[parent] <= keys[i]:
            break
        keys[parent], keys[i] = keys[i], keys[parent]
        vals[parent], vals[i] = vals[i], vals[parent]
        i = parent
    return keys, vals


@njit(cache=True)
def _pop(keys, vals, size):
    """Binary min-heap pop of the root; caller decrements size."""
    key, val = keys[0], vals[0]
    last = size - 1
    keys[0] = keys[last]
    vals[0] = vals[last]
    i = 0
    while True:
        left = 2 * i + 1
        if left >= last:
            break
        child = left
        if left + 1 < last and keys[left + 1] < keys[left]:
            child = left + 1
        if keys[i] <= keys[child]:
            break
        keys[i], keys[child] = keys[child], keys[i]
        vals[i], vals[child] = vals[child], vals[i]
        i = child
    return key, val


@njit(cache=True)
def _search(passable, corner_ok, starts, goal_mask, gy, gx, use_h, dys, dxs, mult, use_w):
    h, w = passable.shape
    n = h * w
    g = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=np.bool_)
    keys = np.empty(1024, dtype=np.float64)
    vals = np.empty(1024, dtype=np.int64)
    size = 0
    for start in starts:  # multi-source: every start begins at cost 0
        g[start] = 0.0
        sy, sx = start // w, start % w
        keys, vals = _push(keys, vals, size, _octile(sy, sx, gy, gx) if use_h else 0.0, start)
        size += 1
    expansions = 0
    while size > 0:
        _, u = _pop(keys, vals, size)
        size -= 1
        if closed[u]:
            continue  # stale heap entry (lazy deletion)
        closed[u] = True
        expansions += 1
        uy, ux = u // w, u % w
        if goal_mask[uy, ux]:
            return parent, g, u, expansions
        for i in range(8):
            vy, vx = uy + dys[i], ux + dxs[i]
            if vy < 0 or vy >= h or vx < 0 or vx >= w or not passable[vy, vx]:
                continue
            if i >= 4:
                if not (corner_ok[vy, ux] and corner_ok[uy, vx]):
                    continue
                step = np.sqrt(2.0)
            else:
                step = 1.0
            if use_w:
                step = step * mult[vy, vx]
            v = vy * w + vx
            if closed[v]:
                continue
            cand = g[u] + step
            if cand < g[v]:
                g[v] = cand
                parent[v] = u
                f = cand + (_octile(vy, vx, gy, gx) if use_h else 0.0)
                keys, vals = _push(keys, vals, size, f, v)
                size += 1
    return parent, g, -1, expansions


@dataclass
class SearchResult:
    path: np.ndarray | None  # int64 [N, 2] (y, x) pixels, start first
    cost: float              # path length in pixels, or weighted cost with
                             # cost_mult (inf if not found)
    expansions: int          # nodes closed: the planner's work, independent of wall time


def search(
    passable: np.ndarray,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    goal_mask: np.ndarray | None = None,
    corner_ok: np.ndarray | None = None,
    start_mask: np.ndarray | None = None,
    cost_mult: np.ndarray | None = None,
) -> SearchResult:
    """A* to `goal` (octile heuristic) or Dijkstra to the nearest pixel of
    `goal_mask` (exactly one of the two), from `start` or from every pixel of
    `start_mask` at once (exactly one of the two). `cost_mult` (float [H, W],
    >= 1 on passable pixels) scales the cost of entering each pixel."""
    if (goal is None) == (goal_mask is None):
        raise ValueError("give exactly one of goal / goal_mask")
    if (start is None) == (start_mask is None):
        raise ValueError("give exactly one of start / start_mask")
    h, w = passable.shape
    if start is not None:
        if not passable[start]:
            raise ValueError(f"start {start} is not passable")
        starts = np.array([start[0] * w + start[1]], dtype=np.int64)
    else:
        starts = np.flatnonzero(start_mask & passable).astype(np.int64)
        if starts.size == 0:
            raise ValueError("start_mask has no passable pixels")
    if corner_ok is None:
        corner_ok = passable
    if cost_mult is None:
        mult, use_w = np.ones((1, 1)), False
    else:
        if cost_mult.shape != (h, w):
            raise ValueError("cost_mult must have the grid's shape")
        mult = np.ascontiguousarray(cost_mult, dtype=np.float64)
        # >= 1 keeps the octile heuristic admissible (NaN fails this too)
        if not (mult[passable] >= 1.0).all():
            raise ValueError("cost_mult must be >= 1 on passable pixels")
        use_w = True
    if goal is not None:
        goal_mask = np.zeros((h, w), dtype=np.bool_)
        goal_mask[goal] = True
        gy, gx, use_h = goal[0], goal[1], True
    else:
        gy, gx, use_h = 0, 0, False
    parent, g, found, expansions = _search(
        np.ascontiguousarray(passable, dtype=np.bool_),
        np.ascontiguousarray(corner_ok, dtype=np.bool_),
        starts,
        np.ascontiguousarray(goal_mask, dtype=np.bool_),
        gy, gx, use_h, _DY, _DX, mult, use_w,
    )
    if found < 0:
        return SearchResult(None, float("inf"), int(expansions))
    nodes = [found]
    while parent[nodes[-1]] >= 0:
        nodes.append(parent[nodes[-1]])
    flat = np.array(nodes[::-1], dtype=np.int64)
    path = np.stack(np.divmod(flat, w), axis=1)
    return SearchResult(path, float(g[found]), int(expansions))


# ---------------------------------------------------------------------------
# Joint roadmap search: entry leg -> roadmap -> exit leg in one A*
# ---------------------------------------------------------------------------


@njit(cache=True)
def _leg_cell(safe, skel, goal, dys, dxs):
    """Pixels a leg can reach from `goal` (same move rules as `_layered`):
    the goal's cell of the roadmap. Restricting the exit layer to it keeps
    the joint search from flooding every cell next to the roadmap."""
    h, w = safe.shape
    cell = np.zeros((h, w), dtype=np.bool_)
    gy, gx = goal // w, goal % w
    if skel[gy, gx]:
        return cell
    stack = np.empty(h * w, dtype=np.int64)
    stack[0] = goal
    top = 1
    cell[gy, gx] = True
    while top > 0:
        top -= 1
        u = stack[top]
        uy, ux = u // w, u % w
        for i in range(8):
            vy, vx = uy + dys[i], ux + dxs[i]
            if vy < 0 or vy >= h or vx < 0 or vx >= w or not safe[vy, vx]:
                continue
            if skel[vy, vx] or cell[vy, vx]:
                continue
            if i >= 4 and not (safe[vy, ux] and safe[uy, vx]):
                continue
            if i >= 4 and skel[vy, ux] and skel[uy, vx]:
                continue
            cell[vy, vx] = True
            stack[top] = vy * w + vx
            top += 1
    return cell


@njit(cache=True)
def _layered(safe, skel, start, goal, leg_mult, road_mult, dys, dxs):
    """A* over three copies of the grid: layer 0 = entry leg, 1 = roadmap,
    2 = exit leg. Legs move through safe non-skeleton pixels and may not
    slip diagonally between two skeleton pixels (that would cross the
    roadmap without joining it). The only way from layer 0 to 1 is stepping
    onto a skeleton pixel, and from 1 to 2 stepping off one."""
    h, w = safe.shape
    n = h * w
    g = np.full(3 * n, np.inf)
    parent = np.full(3 * n, -1, dtype=np.int64)
    closed = np.zeros(3 * n, dtype=np.bool_)
    keys = np.empty(1024, dtype=np.float64)
    vals = np.empty(1024, dtype=np.int64)
    size = 0
    gy, gx = goal // w, goal % w
    s0 = start if not skel[start // w, start % w] else n + start
    g[s0] = 0.0
    keys, vals = _push(keys, vals, size, _octile(start // w, start % w, gy, gx), s0)
    size += 1
    target = (n + goal) if skel[gy, gx] else (2 * n + goal)
    exit_ok = _leg_cell(safe, skel, goal, dys, dxs)
    expansions = 0
    while size > 0:
        _, u = _pop(keys, vals, size)
        size -= 1
        if closed[u]:
            continue
        closed[u] = True
        expansions += 1
        if u == target:
            return parent, g, u, expansions
        layer = u // n
        p = u % n
        uy, ux = p // w, p % w
        for i in range(8):
            vy, vx = uy + dys[i], ux + dxs[i]
            if vy < 0 or vy >= h or vx < 0 or vx >= w or not safe[vy, vx]:
                continue
            on = skel[vy, vx]
            if layer == 0:
                nl = 1 if on else 0
            elif layer == 1:
                nl = 1 if on else 2
            else:
                if on:
                    continue
                nl = 2
            if nl == 2 and not exit_ok[vy, vx]:
                continue
            if i >= 4:
                if not (safe[vy, ux] and safe[uy, vx]):
                    continue
                if nl != 1 and layer != 1 and skel[vy, ux] and skel[uy, vx]:
                    continue  # leg crossing the roadmap diagonally
                step = np.sqrt(2.0)
            else:
                step = 1.0
            # Same accounting as the original planner's separate searches:
            # a leg step costs the multiplier of its pixel nearer the roadmap
            # (the destination on the way in, the source on the way out).
            if nl == 1:
                step = step * (road_mult[vy, vx] if layer == 1 else leg_mult[vy, vx])
            elif nl == 0:
                step = step * leg_mult[vy, vx]
            else:
                step = step * leg_mult[uy, ux]
            v = nl * n + vy * w + vx
            if closed[v]:
                continue
            cand = g[u] + step
            if cand < g[v]:
                g[v] = cand
                parent[v] = u
                keys, vals = _push(keys, vals, size, cand + _octile(vy, vx, gy, gx), v)
                size += 1
    return parent, g, -1, expansions


@dataclass
class LayeredResult:
    path: np.ndarray | None  # int64 [N, 2] (y, x), start first
    layer: np.ndarray | None  # int8 [N]: 0 entry leg, 1 roadmap, 2 exit leg
    cost: float
    expansions: int


def roadmap_search(
    safe: np.ndarray,
    skeleton: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    leg_mult: np.ndarray,
    road_mult: np.ndarray,
) -> LayeredResult:
    """Cheapest start -> (leg) -> roadmap -> (leg) -> goal path, with the join
    and leave pixels chosen jointly instead of each leg ending at its nearest
    skeleton pixel. Legs cost step * leg_mult, roadmap moves step * road_mult
    (both >= 1, so the octile heuristic stays admissible). A leg never
    crosses the skeleton, so it stays in the start's (goal's) own cell of the
    roadmap."""
    h, w = safe.shape
    for name, m in (("leg_mult", leg_mult), ("road_mult", road_mult)):
        if m.shape != (h, w) or not (m[safe] >= 1.0).all():
            raise ValueError(f"{name} must have the grid's shape and be >= 1 on safe pixels")
    if not (safe[start] and safe[goal]):
        raise ValueError("start and goal must be safe")
    skel = np.ascontiguousarray(skeleton & safe, dtype=np.bool_)
    parent, g, found, expansions = _layered(
        np.ascontiguousarray(safe, dtype=np.bool_), skel,
        start[0] * w + start[1], goal[0] * w + goal[1],
        np.ascontiguousarray(leg_mult, dtype=np.float64),
        np.ascontiguousarray(road_mult, dtype=np.float64), _DY, _DX,
    )
    if found < 0:
        return LayeredResult(None, None, float("inf"), int(expansions))
    nodes = [found]
    while parent[nodes[-1]] >= 0:
        nodes.append(parent[nodes[-1]])
    states = np.array(nodes[::-1], dtype=np.int64)
    layer, flat = np.divmod(states, h * w)
    path = np.stack(np.divmod(flat, w), axis=1)
    return LayeredResult(path, layer.astype(np.int8), float(g[found]), int(expansions))
