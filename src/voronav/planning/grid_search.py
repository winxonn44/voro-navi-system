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
