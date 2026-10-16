"""The brute-force oracle is what JFA error is measured against, so it is
itself checked against an independent exact implementation (SciPy's
Euclidean distance transform) and hand-computed tie cases."""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from voronav.maps import random_sites
from voronav.voronoi import NO_SITE, brute_force, distance_sq


@pytest.mark.parametrize("shape,n,seed", [((16, 16), 1, 0), ((33, 47), 12, 1), ((64, 40), 90, 2)])
def test_oracle_distance_matches_scipy_edt(shape, n, seed):
    sites = random_sites(*shape, n, seed=seed)
    d2 = distance_sq(brute_force(sites))
    edt = distance_transform_edt(~sites)
    assert np.array_equal(d2, np.rint(edt**2).astype(np.int64))


def test_oracle_assigns_real_sites_and_sites_to_themselves():
    sites = random_sites(30, 30, 25, seed=5)
    nearest = brute_force(sites)
    assert sites.reshape(-1)[nearest.reshape(-1)].all()
    flat = np.flatnonzero(sites)
    assert np.array_equal(nearest.reshape(-1)[flat], flat)


def test_oracle_tie_break_prefers_smaller_flat_index():
    sites = np.zeros((1, 5), dtype=bool)
    sites[0, 0] = sites[0, 4] = True
    # pixel 2 is equidistant from site 0 and site 4 -> site 0 wins
    assert list(brute_force(sites)[0]) == [0, 0, 0, 4, 4]


def test_oracle_no_sites():
    out = brute_force(np.zeros((4, 6), dtype=bool))
    assert (out == NO_SITE).all()


def test_oracle_chunking_is_invisible(monkeypatch):
    import voronav.voronoi.oracle as oracle

    sites = random_sites(40, 40, 30, seed=7)
    full = oracle.brute_force(sites)
    monkeypatch.setattr(oracle, "_CHUNK_ELEMS", 37)  # force many tiny chunks
    assert np.array_equal(oracle.brute_force(sites), full)
