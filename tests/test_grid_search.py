"""Grid A* / Dijkstra checked against an independent oracle: SciPy's
csgraph Dijkstra on the same 8-connected, no-corner-cutting grid graph."""

import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
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
