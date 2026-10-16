"""Planner contracts, plus the corridor end-to-end test: a map where the
maximum-clearance path is known exactly, so "Voronoi trades length for
clearance" is checked against ground truth rather than eyeballed."""

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
    voronoi_plan,
    weighted_astar_plan,
)

EIGHT = np.ones((3, 3), dtype=bool)


def test_corridor_end_to_end():
    occ = corridor(41, 120, gap=15)  # walls at rows 12 and 28, centerline row 20
    rm = build_roadmap(occ)
    clr = exact_clearance(occ)
    start, goal = (14, 5), (14, 114)  # both hugging the top wall

    vor = voronoi_plan(rm, start, goal)
    ast = astar_plan(rm.safe, start, goal)
    assert vor.success and ast.success
    mv = path_metrics(vor.path, clr, rm.safe)
    ma = path_metrics(ast.path, clr, rm.safe)

    # A* is the shortest path: straight along the wall
    assert ma["length_px"] == pytest.approx(109.0)
    assert ma["mean_clearance_px"] == pytest.approx(2.0)
    # Voronoi leaves the wall, runs the centerline, and comes back
    middle = vor.path[(vor.path[:, 1] >= 20) & (vor.path[:, 1] <= 100)]
    assert set(middle[:, 0].astype(int)) <= {20, 21}
    assert clr[middle[:, 0].astype(int), middle[:, 1].astype(int)].min() >= 7.0
    # the trade-off the whole project measures: longer, but much safer
    assert mv["length_px"] > ma["length_px"]
    assert mv["mean_clearance_px"] > 3 * ma["mean_clearance_px"]
    assert mv["collision_free"] and ma["collision_free"]


def _queries(rm, n, seed):
    regions, _ = label(rm.safe, structure=EIGHT)
    rng = np.random.default_rng(seed)
    pix = np.argwhere(regions == np.bincount(regions.ravel())[1:].argmax() + 1)
    return [(tuple(pix[i]), tuple(pix[j])) for i, j in rng.integers(len(pix), size=(n, 2))]


@pytest.mark.parametrize("seed", range(3))
def test_all_planners_return_valid_paths(seed):
    occ = random_obstacles(96, 96, 8, seed=seed)
    rm = build_roadmap(occ, robot_radius=2.0)
    clr = exact_clearance(occ)
    for start, goal in _queries(rm, 4, seed):
        results = [
            voronoi_plan(rm, start, goal),
            astar_plan(rm.safe, start, goal),
            rrt_star_plan(rm.safe, start, goal, iters=600, seed=seed),
        ]
        for r in results[:2]:  # complete planners: same region -> must succeed
            assert r.success, (r.planner, r.info)
        for r in results:
            if not r.success:
                continue
            assert tuple(r.path[0].astype(int)) == start
            assert tuple(np.rint(r.path[-1]).astype(int)) == goal
            m = path_metrics(r.path, clr, rm.safe)
            assert m["collision_free"], r.planner
            assert m["min_clearance_px"] >= 2.0
        # A* is optimal on the grid, so no grid path is shorter
        assert results[1].path is not None
        la = path_metrics(results[1].path, clr, rm.safe)["length_px"]
        assert path_metrics(results[0].path, clr, rm.safe)["length_px"] >= la - 1e-9


def test_rrt_star_is_reproducible_and_finds_open_paths():
    safe = np.ones((60, 60), dtype=bool)
    a = rrt_star_plan(safe, (5, 5), (55, 55), iters=400, seed=1)
    b = rrt_star_plan(safe, (5, 5), (55, 55), iters=400, seed=1)
    assert a.success and np.array_equal(a.path, b.path)
    # open field: near-straight diagonal, within 10% of optimal
    length = np.hypot(*np.diff(a.path, axis=0).T).sum()
    assert length <= 1.1 * np.hypot(50, 50)


def test_planners_reject_bad_endpoints():
    occ = corridor(21, 40, gap=7)
    rm = build_roadmap(occ)
    with pytest.raises(ValueError):
        voronoi_plan(rm, (0, 0), (10, 30))  # start inside the wall
    with pytest.raises(ValueError):
        astar_plan(rm.safe, (10, 5), (10, 99))  # goal off the map


def test_voronoi_fails_cleanly_without_roadmap():
    rm = build_roadmap(np.zeros((16, 16), dtype=bool))
    r = voronoi_plan(rm, (2, 2), (12, 12))
    assert not r.success and r.info["fail"] == "empty roadmap"


def test_weighted_astar_trades_length_for_clearance():
    occ = random_obstacles(96, 96, 8, seed=1)
    rm = build_roadmap(occ, robot_radius=2.0)
    clr = exact_clearance(occ)
    lengths, clears = [], []
    for start, goal in _queries(rm, 4, 1):
        plain = astar_plan(rm.safe, start, goal)
        w0 = weighted_astar_plan(rm.safe, rm.clearance, start, goal, lam=0.0)
        assert w0.planner == "weighted_astar" and w0.info["lam"] == 0.0
        assert np.array_equal(plain.path, w0.path)  # lam = 0 is plain A*
        row = []
        for lam in (0.0, 8.0, 64.0):
            r = weighted_astar_plan(rm.safe, rm.clearance, start, goal, lam=lam)
            m = path_metrics(r.path, clr, rm.safe)
            assert m["collision_free"]
            row.append(m)
        lengths.append([m["length_px"] for m in row])
        clears.append([m["mean_clearance_px"] for m in row])
    lengths, clears = np.mean(lengths, axis=0), np.mean(clears, axis=0)
    assert lengths[0] <= lengths[1] <= lengths[2]
    assert clears[0] < clears[2]
