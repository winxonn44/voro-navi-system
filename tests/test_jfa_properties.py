"""Properties every JFA backend must satisfy, checked against the oracle.

JFA is approximate, so the core property is one-sided: a JFA pixel can be
assigned a site that is *too far*, never one that is closer than the true
nearest site. Exactness is asserted only where JFA is provably exact (a
single site) and the error rate is pinned on fixed seeds.
"""

import numpy as np
import pytest

from voronav.maps import random_obstacles, boundary_sites, random_sites
from voronav.voronoi import NO_SITE, brute_force, distance_sq, step_schedule, voronoi


def _run(sites, backend, variant="jfa+1"):
    name, device = backend
    return voronoi(sites, backend=name, variant=variant, device=device)


def test_single_site_is_exact(backend):
    # with one site every pixel's only candidate is correct once reached, and
    # the halving schedule reaches every pixel
    sites = np.zeros((37, 53), dtype=bool)
    sites[11, 40] = True
    out = _run(sites, backend, variant="jfa")
    assert (out == 11 * 53 + 40).all()


def test_no_sites_leaves_everything_unassigned(backend):
    out = _run(np.zeros((9, 13), dtype=bool), backend)
    assert (out == NO_SITE).all()


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (23, 1), (2, 3)])
def test_degenerate_shapes(backend, shape):
    sites = np.zeros(shape, dtype=bool)
    sites.flat[-1] = True
    out = _run(sites, backend)
    assert np.array_equal(out, brute_force(sites))


@pytest.mark.parametrize("variant", ["jfa", "jfa+1", "jfa+2", "1+jfa"])
def test_core_invariants(backend, variant):
    sites = random_sites(61, 89, 45, seed=11)
    out = _run(sites, backend, variant)
    ref = brute_force(sites)

    # every pixel is assigned, and only to a real site
    assert (out >= 0).all()
    assert sites.reshape(-1)[out.reshape(-1)].all()
    # sites are their own nearest site (distance 0 cannot be beaten)
    flat = np.flatnonzero(sites)
    assert np.array_equal(out.reshape(-1)[flat], flat)
    # one-sided error: never closer than the true nearest site
    assert (distance_sq(out) >= distance_sq(ref)).all()
    # Note: a correct *distance* does not imply the oracle's *site*. On ties,
    # JFA may never see the smallest-index tied site, so it keeps another
    # equidistant one. That is still a correct Voronoi assignment; error is
    # therefore measured on distance, never on labels.
    ok = distance_sq(out) == distance_sq(ref)
    assert ok.any()


def _error_rate(sites, backend, variant):
    out = _run(sites, backend, variant)
    return float((distance_sq(out) != distance_sq(brute_force(sites))).mean())


def test_error_rate_bounded_and_repaired_by_extra_pass(backend):
    # Pinned on fixed seeds: plain JFA leaves a small residual error that one
    # extra k=1 pass removes on these instances. If this fails after a change,
    # the change altered JFA's behavior, not just its speed.
    totals = {"jfa": 0.0, "jfa+1": 0.0}
    for seed in range(4):
        sites = random_sites(96, 96, 60, seed=seed)
        for v in totals:
            totals[v] += _error_rate(sites, backend, v)
    assert totals["jfa"] < 4 * 0.01
    assert totals["jfa+1"] <= totals["jfa"]


def test_obstacle_map_distance_field(backend):
    occ = random_obstacles(80, 80, 12, seed=2)
    sites = boundary_sites(occ)
    out = _run(sites, backend)
    ref = brute_force(sites)
    err = (distance_sq(out) != distance_sq(ref)).mean()
    assert err < 0.01


def test_step_schedule():
    assert step_schedule(100, 60, "jfa") == [64, 32, 16, 8, 4, 2, 1]
    assert step_schedule(8, 8, "jfa+1") == [4, 2, 1, 1]
    assert step_schedule(8, 8, "jfa+2") == [4, 2, 1, 2, 1]
    assert step_schedule(8, 8, "1+jfa") == [1, 4, 2, 1]
    assert step_schedule(1, 1, "jfa") == []
    with pytest.raises(ValueError):
        step_schedule(8, 8, "jfa+3")
