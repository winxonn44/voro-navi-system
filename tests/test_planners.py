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


BEST = dict(join="best", join_lam=16.0, roadmap_lam=16.0)


def test_best_join_removes_the_corner_spur_detour():
    # One wide room: the GVD is a centerline (row 19-20) plus a diagonal spur
    # from each corner. A start just right of the top-left spur is nearest to
    # the spur, so the original planner backtracks towards the corner, joins
    # the spur and rides it out to the centerline.
    occ = np.ones((40, 124), dtype=bool)
    occ[2:-2, 2:-2] = False
    rm = build_roadmap(occ, robot_radius=2.0)
    clr = exact_clearance(occ)
    start, goal = (5, 15), (20, 110)
    spur = rm.skeleton.copy()
    spur[17:, :] = False  # top-left spur: skeleton pixels above the centerline
    spur[:, 30:] = False
    assert spur.sum() >= 10

    old = voronoi_plan(rm, start, goal)
    new = voronoi_plan(rm, start, goal, **BEST)
    po, pn = old.path.astype(int), new.path.astype(int)
    assert spur[po[:, 0], po[:, 1]].any() and po[:, 1].min() < start[1]  # the detour
    assert not spur[pn[:, 0], pn[:, 1]].any()
    assert pn[:, 1].min() >= start[1]  # never heads back towards the corner
    mo, mn = path_metrics(old.path, clr, rm.safe), path_metrics(new.path, clr, rm.safe)
    assert mn["length_px"] < mo["length_px"] - 5
    assert mn["min_clearance_px"] >= mo["min_clearance_px"]
    assert mn["collision_free"]


def test_best_join_corridor_and_door():
    # the corridor end-to-end claim holds for the new join too
    occ = corridor(41, 120, gap=15)
    rm = build_roadmap(occ)
    clr = exact_clearance(occ)
    r = voronoi_plan(rm, (14, 5), (14, 114), **BEST)
    middle = r.path[(r.path[:, 1] >= 20) & (r.path[:, 1] <= 100)]
    assert set(middle[:, 0].astype(int)) <= {20, 21}
    assert path_metrics(r.path, clr, rm.safe)["collision_free"]
    # and it goes through the door between two rooms
    occ = np.ones((40, 81), dtype=bool)
    occ[2:-2, 2:39] = False
    occ[2:-2, 42:-2] = False
    occ[17:23, 39:42] = False
    rm = build_roadmap(occ)
    r = voronoi_plan(rm, (5, 5), (34, 75), **BEST)
    assert r.success
    door = r.path[(r.path[:, 1] >= 39) & (r.path[:, 1] <= 41)]
    assert len(door) and rm.skeleton[door[:, 0].astype(int), door[:, 1].astype(int)].all()


@pytest.mark.parametrize("seed", range(3))
def test_best_join_legs_stay_off_the_roadmap(seed):
    from voronav.planning import clearance_cost
    from voronav.planning.grid_search import roadmap_search

    occ = random_obstacles(96, 96, 8, seed=seed)
    rm = build_roadmap(occ, robot_radius=2.0)
    leg = clearance_cost(rm.clearance, 16.0)

    def cost(r):  # legs weighted, roadmap unweighted: what both searches minimize
        p = r.path.astype(int)
        n_in, n_out = r.info["entry_px"], r.info["exit_px"]
        length = np.hypot(*np.diff(p, axis=0).T)
        mult = np.ones(len(length))
        mult[:n_in] = leg[p[1:n_in + 1, 0], p[1:n_in + 1, 1]]  # entry: pixel stepped onto
        k = len(length) - n_out
        mult[k:] = leg[p[k:-1, 0], p[k:-1, 1]]  # exit: pixel stepped off
        return (length * mult).sum()

    for start, goal in _queries(rm, 4, seed):
        res = roadmap_search(rm.safe, rm.skeleton, start, goal, leg, np.ones(occ.shape))
        assert res.path is not None
        lay = res.layer
        assert (np.diff(lay) >= 0).all() and (lay == 1).any()  # entry, roadmap, exit in order
        on = rm.skeleton[res.path[:, 0], res.path[:, 1]]
        assert on[lay == 1].all() and not on[lay != 1].any()
        assert np.abs(np.diff(res.path, axis=0)).max() == 1  # 8-connected steps
        # the old planner is one feasible answer of the joint search, so the
        # joint cost under the same weights can only be lower
        old = voronoi_plan(rm, start, goal, join_lam=16.0)
        new = voronoi_plan(rm, start, goal, join="best", join_lam=16.0)
        assert new.success and old.success
        assert cost(new) <= cost(old) + 1e-9
        assert cost(new) == pytest.approx(res.cost)


def test_voronoi_rejects_unknown_join():
    rm = build_roadmap(corridor(21, 40, gap=7))
    with pytest.raises(ValueError):
        voronoi_plan(rm, (10, 5), (10, 30), join="teleport")
