"""Path smoothing: never collides (checked independently by path_metrics on
the exact EDT), never lengthens, and `preserve` never gives back minimum
clearance."""

import numpy as np
import pytest
from scipy.ndimage import label

from voronav.maps import corridor, random_obstacles
from voronav.planning import (
    astar_plan,
    build_roadmap,
    exact_clearance,
    path_metrics,
    rrt_star_plan,
    smooth_path,
    voronoi_plan,
    weighted_astar_plan,
)

EIGHT = np.ones((3, 3), dtype=bool)


def _queries(rm, n, seed):
    regions, _ = label(rm.safe, structure=EIGHT)
    rng = np.random.default_rng(seed)
    pix = np.argwhere(regions == np.bincount(regions.ravel())[1:].argmax() + 1)
    return [(tuple(pix[i]), tuple(pix[j])) for i, j in rng.integers(len(pix), size=(n, 2))]


@pytest.mark.parametrize("seed", range(4))
def test_smoothing_never_collides_or_lengthens(seed):
    occ = random_obstacles(96, 96, 10, seed=seed)
    rm = build_roadmap(occ, robot_radius=2.0)
    clr = exact_clearance(occ)
    for start, goal in _queries(rm, 5, seed):
        results = [
            voronoi_plan(rm, start, goal),
            voronoi_plan(rm, start, goal, join="best", join_lam=16, roadmap_lam=16),
            astar_plan(rm.safe, start, goal),
            weighted_astar_plan(rm.safe, rm.clearance, start, goal, 16.0),
            rrt_star_plan(rm.safe, start, goal, iters=300, seed=seed),
        ]
        for r in results:
            if not r.success:
                continue
            raw = path_metrics(r.path, clr, rm.safe)
            for mode in ("safe", "preserve"):
                sm = smooth_path(r.path, rm.safe, rm.clearance, mode)
                assert np.array_equal(sm[0], r.path[0]) and np.allclose(sm[-1], r.path[-1])
                m = path_metrics(sm, clr, rm.safe)
                assert m["collision_free"], (r.planner, mode)
                assert m["min_clearance_px"] >= rm.robot_radius
                assert m["length_px"] <= raw["length_px"] + 1e-9
                if mode == "preserve":
                    assert m["min_clearance_px"] >= raw["min_clearance_px"] - 1e-9


def test_smoothing_straightens_a_staircase():
    safe = np.ones((40, 40), dtype=bool)
    stairs = np.array([(i // 2 + i % 2, i // 2) for i in range(60)], dtype=float)  # (0,0),(1,0),(1,1),...
    sm = smooth_path(stairs, safe, np.full(safe.shape, np.inf), "safe")
    assert len(sm) == 2  # one straight segment in open space


def test_safe_smoothing_trades_clearance_and_preserve_does_not():
    occ = corridor(41, 120, gap=15)
    rm = build_roadmap(occ)
    clr = exact_clearance(occ)
    vor = voronoi_plan(rm, (14, 5), (26, 114))  # wall to opposite wall, via the centerline
    raw = path_metrics(vor.path, clr, rm.safe)
    safe_m = path_metrics(smooth_path(vor.path, rm.safe, rm.clearance, "safe"), clr, rm.safe)
    keep_m = path_metrics(smooth_path(vor.path, rm.safe, rm.clearance, "preserve"), clr, rm.safe)
    assert safe_m["length_px"] < raw["length_px"]
    assert safe_m["mean_clearance_px"] < raw["mean_clearance_px"]  # cuts across the corridor
    assert keep_m["min_clearance_px"] >= raw["min_clearance_px"]
    assert safe_m["length_px"] <= keep_m["length_px"] <= raw["length_px"] + 1e-9


def test_smoothing_rejects_unknown_mode():
    with pytest.raises(ValueError):
        smooth_path(np.zeros((3, 2)), np.ones((4, 4), bool), np.ones((4, 4)), "bezier")
