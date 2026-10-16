"""Every parallel backend must be bit-identical to the NumPy reference, for
every variant and any thread count. This is what lets the Q1 study report
speedups without an accuracy caveat: the fast paths compute the same thing."""

import numpy as np
import pytest

from voronav.maps import boundary_sites, random_obstacles, random_sites
from voronav.voronoi import voronoi
from voronav.voronoi.jfa_numba import max_threads

CASES = [
    ("points-sparse", lambda: random_sites(128, 128, 8, seed=0)),
    ("points-dense", lambda: random_sites(100, 150, 600, seed=1)),
    ("obstacles", lambda: boundary_sites(random_obstacles(120, 90, 15, seed=2))),
]


@pytest.mark.parametrize("variant", ["jfa", "jfa+1", "1+jfa"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matches_numpy_reference(backend, case, variant):
    sites = case[1]()
    ref = voronoi(sites, backend="numpy", variant=variant)
    name, device = backend
    out = voronoi(sites, backend=name, variant=variant, device=device)
    assert out.dtype == np.int64
    assert np.array_equal(out, ref)


def test_numba_independent_of_thread_count():
    sites = random_sites(150, 150, 200, seed=9)
    outs = [
        voronoi(sites, backend="numba", threads=t)
        for t in sorted({1, 2, max_threads()})
    ]
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
