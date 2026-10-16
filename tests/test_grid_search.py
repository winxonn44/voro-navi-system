"""Grid A* / Dijkstra checked against an independent oracle: SciPy's
csgraph Dijkstra on the same 8-connected, no-corner-cutting grid graph."""

import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra

from voronav.maps import random_obstacles
from voronav.planning import search


def _grid_graph(passable):
    h, w = passable.shape
    rows, cols, wts = [], [], []
    for y in range(h):
        for x in range(w):
            if not passable[y, x]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    vy, vx = y + dy, x + dx
                    if not (0 <= vy < h and 0 <= vx < w) or not passable[vy, vx]:
                        continue
                    if dy and dx and not (passable[vy, x] and passable[y, vx]):
                        continue
                    rows.append(y * w + x)
                    cols.append(vy * w + vx)
                    wts.append(math.sqrt(2.0) if dy and dx else 1.0)
    return coo_matrix((wts, (rows, cols)), shape=(h * w, h * w)).tocsr()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_astar_cost_matches_scipy_dijkstra(seed):
    occ = random_obstacles(40, 40, 6, seed=seed)
    passable = ~occ
    rng = np.random.default_rng(seed)
    free = np.argwhere(passable)
    dist = dijkstra(_grid_graph(passable), indices=None, directed=True)
    w = passable.shape[1]
    for _ in range(10):
        a, b = (tuple(free[i]) for i in rng.integers(len(free), size=2))
        expected = dist[a[0] * w + a[1], b[0] * w + b[1]]
        res = search(passable, a, goal=b)
        if math.isinf(expected):
            assert res.path is None
        else:
            assert res.cost == pytest.approx(expected)
            # the reconstructed path really has that length and is legal
            steps = np.diff(res.path, axis=0)
            assert (np.abs(steps).max(axis=1) == 1).all()
            assert np.hypot(steps[:, 0], steps[:, 1]).sum() == pytest.approx(expected)
            assert passable[res.path[:, 0], res.path[:, 1]].all()


def test_no_corner_cutting():
    passable = np.array([[1, 0], [0, 1]], dtype=bool)  # diagonal through a corner
    assert search(passable, (0, 0), goal=(1, 1)).path is None


def test_dijkstra_to_nearest_target():
    passable = np.ones((1, 10), dtype=bool)
    targets = np.zeros((1, 10), dtype=bool)
    targets[0, 2] = targets[0, 9] = True
    res = search(passable, (0, 5), goal_mask=targets)
    assert tuple(res.path[-1]) == (0, 2) and res.cost == 3


def test_unreachable_and_bad_start():
    passable = np.ones((5, 5), dtype=bool)
    passable[:, 2] = False
    assert search(passable, (0, 0), goal=(0, 4)).path is None
    with pytest.raises(ValueError):
        search(passable, (0, 2), goal=(0, 4))
    with pytest.raises(ValueError):
        search(passable, (0, 0))


def _weighted_graph(passable, mult):
    """Same graph, each edge scaled by the multiplier of the pixel it enters."""
    g = _grid_graph(passable).tocoo()
    scale = mult.reshape(-1)[g.col]
    return coo_matrix((g.data * scale, (g.row, g.col)), shape=g.shape).tocsr()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_astar_cost_matches_scipy_dijkstra(seed):
    occ = random_obstacles(40, 40, 6, seed=seed)
    passable = ~occ
    clr = distance_transform_edt(passable)
    mult = np.where(passable, 1.0 + 4.0 / np.maximum(clr, 1e-9), 1.0)
    rng = np.random.default_rng(seed)
    free = np.argwhere(passable)
    dist = dijkstra(_weighted_graph(passable, mult), indices=None, directed=True)
    w = passable.shape[1]
    for _ in range(10):
        a, b = (tuple(free[i]) for i in rng.integers(len(free), size=2))
        expected = dist[a[0] * w + a[1], b[0] * w + b[1]]
        res = search(passable, a, goal=b, cost_mult=mult)
        if math.isinf(expected):
            assert res.path is None
            continue
        assert res.cost == pytest.approx(expected)
        steps = np.diff(res.path, axis=0)
        cost = (np.hypot(steps[:, 0], steps[:, 1]) * mult[res.path[1:, 0], res.path[1:, 1]]).sum()
        assert cost == pytest.approx(expected)


def test_unit_weights_match_unweighted_exactly():
    occ = random_obstacles(48, 48, 8, seed=3)
    passable = ~occ
    free = np.argwhere(passable)
    rng = np.random.default_rng(3)
    for _ in range(5):
        a, b = (tuple(free[i]) for i in rng.integers(len(free), size=2))
        plain = search(passable, a, goal=b)
        ones = search(passable, a, goal=b, cost_mult=np.ones(passable.shape))
        assert plain.cost == ones.cost and plain.expansions == ones.expansions
        assert (plain.path is None) == (ones.path is None)
        if plain.path is not None:
            assert np.array_equal(plain.path, ones.path)


def test_cost_mult_below_one_rejected():
    passable = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match=">= 1"):
        search(passable, (0, 0), goal=(3, 3), cost_mult=np.full((4, 4), 0.5))
