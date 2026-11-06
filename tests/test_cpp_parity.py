"""Cross-language parity: the C++ core (voronav_core) against the Python
reference, compared with array_equal on the `nearest` arrays themselves, not
just on distances. Two implementations that agree only on distance could
still break ties differently; equal site indices rule that out.

Skipped as a whole when the core is not installed (`pip install -e ./core`).
"""

import numpy as np
import pytest

from voronav.config import VARIANTS
from voronav.voronoi import brute_force, distance_sq, init_nearest, jfa_numpy, step_schedule

core = pytest.importorskip("voronav_core")

# Degenerate, odd, non-square and power-of-two-boundary shapes. Sides 1-2
# have zero or one standard step; 31/33 and 63/65 straddle a power of two.
SHAPES = [
    (1, 1), (1, 2), (2, 1), (1, 37), (41, 1), (2, 3), (3, 3), (5, 8),
    (12, 13), (16, 16), (17, 15), (31, 33), (33, 31), (40, 64), (63, 65), (29, 90),
]
DENSITIES = [0.0, 0.003, 0.02, 0.1, 0.4, 1.0]
SEEDS = range(6)


def _sites(shape, density, seed):
    rng = np.random.default_rng([seed, *shape, int(density * 1000)])
    return rng.random(shape) < density


def _assert_same(sites, variant):
    steps = step_schedule(*sites.shape, variant)
    init = init_nearest(sites)
    ref = jfa_numpy.run_steps(init, steps)
    out = core.run_steps(core.init_nearest(sites), steps)
    assert out.dtype == np.int64 and out.shape == ref.shape
    assert np.array_equal(out, ref), (sites.shape, variant, np.argwhere(out != ref)[:5])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_run_steps_matches_numpy(shape, variant):
    """6 densities x 6 seeds per (shape, variant): 576 grids per variant."""
    for density in DENSITIES:
        for seed in SEEDS:
            _assert_same(_sites(shape, density, seed), variant)


@pytest.mark.parametrize("seed", range(8))
def test_run_steps_matches_numpy_on_random_shapes(seed):
    """Shapes drawn at random too, so no size is special-cased by the list."""
    rng = np.random.default_rng(1000 + seed)
    for _ in range(10):
        h, w = (int(v) for v in rng.integers(1, 80, size=2))
        density = float(rng.choice([0.001, 0.01, 0.05, 0.3]))
        sites = rng.random((h, w)) < density
        for variant in VARIANTS:
            _assert_same(sites, variant)


# The thread pool (checkpoint C3): every thread count, schedule and chunk must
# give the NumPy result bit for bit. Shapes include fewer rows than threads.
THREADS = [2, 3, 7, 10, None]
THREAD_SHAPES = [(1, 1), (1, 37), (41, 1), (3, 50), (12, 13), (33, 31), (64, 64), (97, 131)]


@pytest.mark.parametrize("schedule", ["static", "dynamic"])
@pytest.mark.parametrize("threads", THREADS, ids=lambda t: f"t{t}")
def test_threaded_run_steps_matches_numpy(threads, schedule):
    for shape in THREAD_SHAPES:
        for density in (0.0, 0.01, 0.1, 1.0):
            sites = _sites(shape, density, seed=3)
            init = init_nearest(sites)
            for variant in VARIANTS:
                steps = step_schedule(*shape, variant)
                ref = jfa_numpy.run_steps(init, steps)
                for chunk in (1, 4, 16):
                    out = core.run_steps(init, steps, threads, schedule, chunk)
                    assert np.array_equal(out, ref), (shape, density, variant, threads, chunk)


def test_threaded_large_grid_matches_numpy():
    sites = _sites((512, 384), 0.002, seed=4)
    init = init_nearest(sites)
    steps = step_schedule(512, 384, "jfa+2")
    ref = jfa_numpy.run_steps(init, steps)
    for threads in (4, 10):
        for schedule in ("static", "dynamic"):
            assert np.array_equal(core.run_steps(init, steps, threads, schedule), ref)


def test_lattice_ties_match_numpy():
    """Many pixels equidistant from 2 or 4 sites: exercises the tie-break."""
    sites = np.zeros((97, 131), dtype=bool)
    sites[3::8, 5::8] = True
    for variant in VARIANTS:
        _assert_same(sites, variant)


# The minimal jfa+2 counterexample found by C1's exhaustive two-site search
# (core/tests/test_oracle.cpp, OracleAgreement.JfaPlus2MinimalCounterexample):
# a 12 x 13 grid with sites 39 = (3, 0) and 151 = (11, 8). Every variant except
# 1+jfa leaves pixels on site 39 that belong to site 151.
COUNTEREXAMPLE_WRONG = {
    "jfa": [12, 24, 25, 36, 37, 38, 48, 49, 50, 51],
    "jfa+1": [12, 24, 25, 36, 37, 38],
    "jfa+2": [12],
    "1+jfa": [],
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_pinned_counterexample_is_wrong_identically_in_both_languages(variant):
    """The parity anchor: where JFA is wrong, C++ must be wrong in exactly the
    same way as Python, pixel for pixel and site for site."""
    sites = np.zeros((12, 13), dtype=bool)
    sites.flat[[39, 151]] = True
    steps = step_schedule(12, 13, variant)
    py_out = jfa_numpy.run_steps(init_nearest(sites), steps)
    cpp_out = core.run_steps(core.init_nearest(sites), steps)
    assert np.array_equal(cpp_out, py_out)
    for threads in (2, 7, 10):
        for schedule in ("static", "dynamic"):
            par = core.run_steps(core.init_nearest(sites), steps, threads, schedule, 1)
            assert np.array_equal(par, py_out)

    oracle = brute_force(sites)
    assert np.array_equal(core.brute_force(sites), oracle)
    wrong = np.flatnonzero(distance_sq(py_out) != distance_sq(oracle))
    assert wrong.tolist() == COUNTEREXAMPLE_WRONG[variant]
    assert (cpp_out.flat[wrong] == 39).all() and (oracle.flat[wrong] == 151).all()
    if variant == "jfa+2":
        assert distance_sq(oracle).flat[12] == 137 and distance_sq(cpp_out).flat[12] == 153


def test_step_schedule_matches_python():
    sides = list(range(0, 70)) + [127, 128, 129, 1000, 1023, 1024, 1025, 4096, 46341, 2**20 + 1]
    for variant in VARIANTS:
        for h in sides:
            for w in (1, 2, 3, 7, 64, 65, h, 2**16):
                assert core.step_schedule(h, w, variant) == step_schedule(h, w, variant), (
                    h, w, variant)
    assert core.step_schedule(8, 8) == step_schedule(8, 8)  # same default variant


@pytest.mark.parametrize("shape", [(1, 1), (1, 37), (12, 13), (33, 31), (29, 90)],
                         ids=lambda s: f"{s[0]}x{s[1]}")
def test_init_distance_and_oracle_match_python(shape):
    for density in DENSITIES:
        for seed in SEEDS:
            sites = _sites(shape, density, seed)
            init = init_nearest(sites)
            assert np.array_equal(core.init_nearest(sites), init)
            assert np.array_equal(core.init_nearest(sites.view(np.uint8)), init)
            oracle = brute_force(sites)
            assert np.array_equal(core.brute_force(sites), oracle)
            assert np.array_equal(core.distance_sq(oracle), distance_sq(oracle))
            assert np.array_equal(core.distance_sq(init), distance_sq(init))


def test_uint8_sites_treat_any_nonzero_byte_as_a_site():
    sites = np.zeros((5, 6), dtype=np.uint8)
    sites[1, 2], sites[3, 4] = 1, 255
    assert np.array_equal(core.init_nearest(sites), init_nearest(sites.astype(bool)))
    assert np.array_equal(core.brute_force(sites), brute_force(sites.astype(bool)))


# The C4 kernels: every kernel the core can be told to use gives the NumPy
# result. 'auto' (the default, used by every test above) is 'packed' at these
# sizes; 'split' is its fallback above 32767 px per side.
KERNELS = ["reference", "split", "packed", "auto"]


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_matches_numpy(kernel):
    # Sides 1-6 have no interior at most steps; the rest mix both.
    shapes = [(h, w) for h in (1, 2, 3, 5, 6) for w in (1, 2, 4, 6)]
    shapes += [(12, 13), (31, 33), (2, 90), (90, 3), (64, 64), (97, 131)]
    for shape in shapes:
        for density in (0.0, 0.01, 0.1, 1.0):
            sites = _sites(shape, density, seed=5)
            init = init_nearest(sites)
            for variant in VARIANTS:
                steps = step_schedule(*shape, variant)
                ref = jfa_numpy.run_steps(init, steps)
                for threads in (1, 4):
                    out = core.run_steps(init, steps, threads, "dynamic", 4, kernel)
                    assert np.array_equal(out, ref), (shape, density, variant, threads, kernel)


def test_kernel_fallback_above_the_packed_limit():
    assert core.resolve_kernel(32767, 32767) == "packed"
    assert core.resolve_kernel(1, 32768) == "split"
    assert core.resolve_kernel(32768, 1, "reference") == "reference"
    sites = np.zeros((1, 32768), dtype=bool)
    sites[0, [0, 5000, 32767]] = True
    init = init_nearest(sites)
    steps = step_schedule(1, 32768, "jfa+1")
    ref = jfa_numpy.run_steps(init, steps)
    assert np.array_equal(core.run_steps(init, steps), ref)  # auto -> split
    with pytest.raises(ValueError, match="32767"):
        core.run_steps(init, steps, kernel="packed")
    with pytest.raises(ValueError, match="kernel must be"):
        core.run_steps(init, steps, kernel="simd")
