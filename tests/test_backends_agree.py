"""Every parallel backend must be bit-identical to the NumPy reference, for
every variant and any thread count. This is what lets the Q1 study report
speedups without an accuracy caveat: the fast paths compute the same thing."""

import numpy as np
import pytest
import torch

from voronav.maps import boundary_sites, random_obstacles, random_sites
from voronav.voronoi import init_nearest, step_schedule, voronoi
from voronav.voronoi import jfa_numba, jfa_torch
from voronav.voronoi.jfa_numba import max_threads
from voronav.voronoi.jfa_torch import available_devices

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


def test_numba_independent_of_thread_count_and_schedule():
    sites = random_sites(150, 150, 200, seed=9)
    ref = voronoi(sites, backend="numpy")
    for schedule in ("static", "dynamic"):
        for t in sorted({1, 2, 3, max_threads()}):
            out = voronoi(sites, backend="numba", threads=t, schedule=schedule)
            assert np.array_equal(out, ref), (schedule, t)


def test_numba_dynamic_chunk_sizes():
    """Chunks that do not divide the height, and chunks larger than it."""
    sites = random_sites(37, 53, 40, seed=3)
    init = init_nearest(sites)
    steps = step_schedule(*sites.shape, "jfa+1")
    ref = voronoi(sites, backend="numpy")
    for chunk in (1, 5, 64):
        out = jfa_numba.run_steps(init, steps, threads=max_threads(), chunk=chunk)
        assert np.array_equal(out, ref), chunk


def _lattice_sites():
    """Sites on a regular lattice: many pixels are exactly equidistant from
    two or four sites, so this exercises the tie-break, not just distance."""
    sites = np.zeros((97, 131), dtype=bool)
    sites[3::8, 5::8] = True
    return sites


@pytest.mark.skipif("mps" not in available_devices(), reason="needs MPS")
@pytest.mark.parametrize("fused", [True, False], ids=["metal", "eager"])
@pytest.mark.parametrize("variant", ["jfa", "jfa+2"])
@pytest.mark.parametrize(
    "make_sites",
    [CASES[1][1], CASES[2][1], _lattice_sites, lambda: random_sites(1, 37, 3, seed=4)],
    ids=["points-dense", "obstacles", "lattice-ties", "1xN"],
)
def test_mps_paths_match_reference(make_sites, variant, fused):
    """Both MPS code paths (the int32 Metal kernel and the int64 eager ops)
    must equal the NumPy reference, including on exact ties."""
    sites = make_sites()
    ref = voronoi(sites, backend="numpy", variant=variant)
    init = jfa_torch.to_device(init_nearest(sites), "mps")
    out = jfa_torch.run_steps(init, step_schedule(*sites.shape, variant), fused=fused)
    assert out.dtype == torch.int64
    assert np.array_equal(out.cpu().numpy(), ref)
