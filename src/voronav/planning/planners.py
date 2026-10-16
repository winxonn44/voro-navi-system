"""The planners compared in Q2.

  voronoi: get onto the GVD roadmap, travel along it, get off at the goal.
           Maximum clearance by construction.
  astar:   shortest 8-connected path through safe free space. The baseline
           that the clearance claim is measured against.
  weighted_astar: grid A* where entering a pixel costs 1 + lambda / clearance
           per unit length. A fairer baseline than plain A*, which hugs walls
           by construction: sweeping lambda trades length for clearance.
  rrt_star: sampling-based asymptotically optimal planner (Karaman & Frazzoli
           2011), continuous space. A path-quality baseline only: it is
           pure Python/NumPy, so its wall time is not comparable to the
           compiled grid planners.

All planners take the same `safe` mask (free space eroded by the robot
radius), so a path from any of them is collision-free for the same robot.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .grid_search import search
from .roadmap import Roadmap


@dataclass
class PlanResult:
    planner: str
    path: np.ndarray | None  # float64 [N, 2] (y, x), start first; None on failure
    plan_s: float            # query time (roadmap build time is reported separately)
    expansions: int = 0
    info: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.path is not None


def _check_endpoints(safe: np.ndarray, start, goal) -> None:
    for name, p in (("start", start), ("goal", goal)):
        y, x = p
        if not (0 <= y < safe.shape[0] and 0 <= x < safe.shape[1]) or not safe[y, x]:
            raise ValueError(f"{name} {p} is not in safe free space")


# ---------------------------------------------------------------------------
# Voronoi roadmap
# ---------------------------------------------------------------------------


def voronoi_plan(rm: Roadmap, start: tuple[int, int], goal: tuple[int, int]) -> PlanResult:
    """Connect start and goal to the nearest skeleton pixel (multi-target
    Dijkstra through safe space), then A* along the skeleton only."""
    _check_endpoints(rm.safe, start, goal)
    t0 = time.perf_counter()
    expansions = 0

    def fail(reason: str) -> PlanResult:
        return PlanResult("voronoi", None, time.perf_counter() - t0, expansions, {"fail": reason})

    if not rm.skeleton.any():
        return fail("empty roadmap")
    entry = search(rm.safe, start, goal_mask=rm.skeleton)
    exit_ = search(rm.safe, goal, goal_mask=rm.skeleton)
    expansions += entry.expansions + exit_.expansions
    if entry.path is None or exit_.path is None:
        return fail("cannot reach roadmap")
    on = tuple(entry.path[-1])
    off = tuple(exit_.path[-1])
    core = search(rm.skeleton, on, goal=off, corner_ok=rm.safe)
    expansions += core.expansions
    if core.path is None:
        return fail("roadmap disconnected")
    path = np.concatenate([entry.path, core.path[1:], exit_.path[::-1][1:]]).astype(np.float64)
    return PlanResult(
        "voronoi",
        path,
        time.perf_counter() - t0,
        expansions,
        {"entry_px": len(entry.path) - 1, "exit_px": len(exit_.path) - 1},
    )


# ---------------------------------------------------------------------------
# Grid A*
# ---------------------------------------------------------------------------


def astar_plan(safe: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> PlanResult:
    _check_endpoints(safe, start, goal)
    t0 = time.perf_counter()
    res = search(safe, start, goal=goal)
    path = None if res.path is None else res.path.astype(np.float64)
    return PlanResult("astar", path, time.perf_counter() - t0, res.expansions)


def clearance_cost(clearance: np.ndarray, lam: float) -> np.ndarray:
    """Step-cost multiplier 1 + lam / clearance (>= 1; 1 where clearance is
    inf, i.e. no obstacles; inf on obstacles, which are never entered)."""
    if lam < 0:
        raise ValueError("lam must be >= 0")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(clearance > 0, 1.0 + lam / clearance, np.inf)


def weighted_astar_plan(
    safe: np.ndarray,
    clearance: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    lam: float,
) -> PlanResult:
    """Clearance-weighted grid A*: step cost = step length * (1 + lam /
    clearance of the destination pixel). lam = 0 is plain A*. The cost map is
    built inside the timed query (one O(H*W) NumPy pass), like A*'s own
    per-query arrays."""
    _check_endpoints(safe, start, goal)
    t0 = time.perf_counter()
    res = search(safe, start, goal=goal, cost_mult=clearance_cost(clearance, lam))
    path = None if res.path is None else res.path.astype(np.float64)
    return PlanResult("weighted_astar", path, time.perf_counter() - t0, res.expansions,
                      {"lam": lam, "weighted_cost": res.cost})


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------


def segment_free(safe: np.ndarray, a: np.ndarray, b: np.ndarray, spacing: float = 0.25) -> bool:
    """Whether the straight segment a->b stays in `safe`, checked at the
    nearest pixel every `spacing` px. The spacing matches `metrics.resample`:
    at 0.5 px the planner skipped corner pixels that the independent
    collision check then caught."""
    n = max(2, int(math.ceil(np.hypot(*(b - a)) / spacing)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.rint(a + t * (b - a)).astype(np.int64)
    h, w = safe.shape
    if (pts < 0).any() or (pts[:, 0] >= h).any() or (pts[:, 1] >= w).any():
        return False
    return bool(safe[pts[:, 0], pts[:, 1]].all())


def rrt_star_plan(
    safe: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    iters: int = 2000,
    step: float = 8.0,
    goal_bias: float = 0.05,
    seed: int = 0,
) -> PlanResult:
    """RRT* with a fixed iteration budget; returns the best path found.

    Rewiring radius r = min(gamma * sqrt(log n / n), 3 * step), with gamma
    set from the free area as in Karaman & Frazzoli (2011), Theorem 38."""
    _check_endpoints(safe, start, goal)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    free_px = np.argwhere(safe)
    gamma = 2.0 * math.sqrt(1.5) * math.sqrt(len(free_px) / math.pi) * 1.1
    goal_pt = np.array(goal, dtype=np.float64)

    cap = iters + 1
    pos = np.empty((cap, 2))
    parent = np.full(cap, -1, dtype=np.int64)
    cost = np.empty(cap)
    pos[0] = start
    cost[0] = 0.0
    n = 1

    for _ in range(iters):
        if rng.random() < goal_bias:
            sample = goal_pt
        else:
            sample = free_px[rng.integers(len(free_px))] + rng.uniform(-0.5, 0.5, 2)
        d = np.hypot(*(pos[:n] - sample).T)
        near_i = int(np.argmin(d))
        direction = sample - pos[near_i]
        dist = d[near_i]
        if dist < 1e-9:
            continue
        new = pos[near_i] + direction * min(1.0, step / dist)
        if not segment_free(safe, pos[near_i], new):
            continue

        r = min(gamma * math.sqrt(math.log(n + 1) / (n + 1)), 3.0 * step)
        dn = np.hypot(*(pos[:n] - new).T)
        near = np.flatnonzero(dn <= r)
        # choose the parent that gives the cheapest collision-free arrival
        best_p, best_c = near_i, cost[near_i] + np.hypot(*(new - pos[near_i]))
        for j in near[np.argsort(cost[near] + dn[near])]:
            c = cost[j] + dn[j]
            if c >= best_c:
                break  # sorted: no later candidate can be cheaper
            if segment_free(safe, pos[j], new):
                best_p, best_c = int(j), c
        k = n
        pos[k], parent[k], cost[k] = new, best_p, best_c
        n += 1
        # rewire neighbors through the new node when that is cheaper
        for j in near:
            c = best_c + dn[j]
            if c < cost[j] and segment_free(safe, new, pos[j]):
                delta = c - cost[j]
                parent[j], cost[j] = k, c
                # propagate the improvement to j's descendants
                stack = [int(j)]
                while stack:
                    u = stack.pop()
                    kids = np.flatnonzero(parent[:n] == u)
                    cost[kids] += delta
                    stack.extend(int(v) for v in kids)

    # Connect to the goal only at the end: rewiring keeps lowering node costs,
    # so the best goal parent is only known once the budget is spent.
    dg = np.hypot(*(pos[:n] - goal_pt).T)
    cand = np.flatnonzero(dg <= step)
    best = -1
    for j in cand[np.argsort(cost[cand] + dg[cand])]:
        if segment_free(safe, pos[j], goal_pt):
            best = int(j)
            break
    if best < 0:
        return PlanResult("rrt_star", None, time.perf_counter() - t0, n, {"nodes": n})
    chain = [best]
    while parent[chain[-1]] >= 0:
        chain.append(int(parent[chain[-1]]))
    path = np.vstack([pos[chain[::-1]], goal_pt[None, :]])
    return PlanResult(
        "rrt_star", path, time.perf_counter() - t0, n,
        {"nodes": n, "cost": float(cost[best] + dg[best])},
    )
