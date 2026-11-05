"""The optional C++ core (`core/`, distribution `voronav-core`). Tests that
need the compiled module skip when it is not installed, so the main suite
stays green after a plain `pip install -e .`."""

import importlib.metadata
import sys

import numpy as np
import pytest

from voronav.maps import random_sites
from voronav.voronoi import make_runner


def test_cpp_backend_without_core_names_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "voronav_core", None)  # import now fails
    with pytest.raises(ImportError, match=r"pip install -e \./core"):
        make_runner(random_sites(8, 8, 2, seed=0), backend="cpp")


@pytest.fixture(scope="module")
def core():
    return pytest.importorskip("voronav_core")


def test_core_version_matches_package_metadata(core):
    assert core.version() == importlib.metadata.version("voronav-core")
    assert core.__version__ == core.version()


def test_cpp_backend_runs_through_make_runner(core):
    sites = random_sites(19, 23, 5, seed=0)
    run, fetch = make_runner(sites, backend="cpp", variant="jfa+2")
    run()
    ref_run, ref_fetch = make_runner(sites, backend="numpy", variant="jfa+2")
    ref_run()
    assert np.array_equal(fetch(), ref_fetch())


@pytest.mark.parametrize(
    "kwargs",
    [{"threads": 0}, {"threads": -2}, {"schedule": "guided"}, {"device": "mps"}],
    ids=["threads0", "threads-2", "guided", "mps"],
)
def test_cpp_backend_rejects_bad_settings_at_setup(core, kwargs):
    with pytest.raises(ValueError, match="backend='cpp'"):
        make_runner(random_sites(8, 8, 2, seed=0), backend="cpp", **kwargs)


@pytest.mark.parametrize("threads", [None, 1, 3, 10])
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
def test_cpp_backend_threads_and_schedule(core, threads, schedule):
    sites = random_sites(37, 29, 6, seed=1)
    run, fetch = make_runner(sites, backend="cpp", variant="jfa+2", threads=threads,
                             schedule=schedule)
    run()
    ref_run, ref_fetch = make_runner(sites, backend="numpy", variant="jfa+2")
    ref_run()
    assert np.array_equal(fetch(), ref_fetch())


def test_core_rejects_bad_threads_schedule_chunk(core):
    init = np.full((4, 5), -1, dtype=np.int64)
    for kwargs in ({"threads": 0}, {"threads": 1025}, {"schedule": "guided"}, {"chunk": 0}):
        with pytest.raises(ValueError):
            core.run_steps(init, [2, 1], **kwargs)


def test_pools_are_cached_per_thread_count(core):
    """One pool per thread count, reused by later calls; threads=1 is the
    serial path and creates none."""
    pool_sizes = core._core._pool_sizes
    init = core.init_nearest(np.eye(16, dtype=bool))
    steps = core.step_schedule(16, 16, "jfa+1")
    core.run_steps(init, steps, threads=1)
    assert 1 not in pool_sizes()
    for _ in range(20):
        core.run_steps(init, steps, threads=5, schedule="static")
        core.run_steps(init, steps, threads=5, schedule="dynamic", chunk=1)
    assert pool_sizes().count(5) == 1
    core.run_steps(init, steps, threads=None)
    assert core.hardware_threads() in pool_sizes() or core.hardware_threads() == 1


def test_run_steps_releases_the_gil(core):
    """While one thread is inside C++ run_steps, another keeps running Python:
    with the GIL held for the call, the counter below could not advance."""
    import threading

    init = core.init_nearest(np.eye(1024, dtype=bool))
    steps = core.step_schedule(1024, 1024, "jfa+2") * 3  # ~0.2 s serial
    started = threading.Event()

    def work():
        started.set()
        core.run_steps(init, steps, threads=1)

    t = threading.Thread(target=work)
    t.start()
    started.wait()
    ticks = 0
    while t.is_alive():
        ticks += 1
    t.join()
    assert ticks > 1000


def test_one_pool_shared_by_concurrent_callers(core):
    """Four Python threads call the same cached pool at once (the GIL is
    released, so the calls overlap in C++); every result is still exact."""
    import threading

    from voronav.voronoi import init_nearest, jfa_numpy, step_schedule

    sites = random_sites(256, 256, 60, seed=2)
    init, steps = init_nearest(sites), step_schedule(256, 256, "jfa+2")
    ref = jfa_numpy.run_steps(init, steps)
    results = [None] * 4

    def work(i):
        results[i] = core.run_steps(init, steps, threads=4, schedule="dynamic")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(np.array_equal(r, ref) for r in results)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64])
def test_jfa_inputs_cross_zero_copy(core, dtype):
    # data_address goes through the same checks as init_nearest / run_steps
    a = np.zeros((6, 9), dtype=dtype)
    assert core.data_address(a) == a.ctypes.data


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((4, 6), dtype=np.int64)[:, ::2], "C-contiguous"),
        (np.zeros((4, 6), dtype=np.int64, order="F"), "C-contiguous"),
        (np.zeros((4, 6), dtype=np.int32), "int64"),
        (np.zeros((4, 6), dtype=">i8"), "int64"),  # byte-swapped: would need a copy
        (np.zeros(6, dtype=np.int64), "2D"),
        (np.zeros((2, 3, 4), dtype=np.int64), "2D"),
        ([[0, -1]], "NumPy array"),
        (np.zeros(65, dtype=np.uint8)[1:].view(np.int64).reshape(2, 4), "aligned"),
    ],
    ids=["strided", "fortran", "int32", "big-endian", "1d", "3d", "list", "misaligned"],
)
def test_run_steps_rejects_arrays_that_would_be_copied(core, bad, match):
    with pytest.raises(ValueError, match=match):
        core.run_steps(bad, [1])
    with pytest.raises(ValueError, match=match):
        core.distance_sq(bad)


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((4, 6), dtype=bool)[:, ::2], "C-contiguous"),
        (np.zeros((4, 6), dtype=np.int64), "bool or uint8"),
        (np.zeros((4, 6), dtype=np.int8), "bool or uint8"),
        (np.zeros(6, dtype=bool), "2D"),
    ],
    ids=["strided", "int64", "int8", "1d"],
)
def test_sites_functions_reject_bad_masks(core, bad, match):
    for fn in (core.init_nearest, core.brute_force):
        with pytest.raises(ValueError, match=match):
            fn(bad)


@pytest.mark.parametrize("value", [-2, 12, 2**40])
def test_run_steps_rejects_out_of_range_entries(core, value):
    init = np.full((3, 4), -1, dtype=np.int64)
    init[1, 2] = value
    with pytest.raises(ValueError, match="flat index"):
        core.run_steps(init, [1])
    with pytest.raises(ValueError, match="flat index"):
        core.distance_sq(init)


def test_step_schedule_rejects_unknown_variant(core):
    with pytest.raises(ValueError, match="variant"):
        core.step_schedule(8, 8, "jfa+3")


def test_outputs_own_their_cpp_buffer(core):
    sites = random_sites(7, 11, 3, seed=1)
    init = core.init_nearest(sites)
    outs = [init, core.run_steps(init, [4, 2, 1]), core.distance_sq(init),
            core.brute_force(sites)]
    for out in outs:
        assert out.dtype == np.int64 and out.shape == (7, 11)
        assert out.flags.c_contiguous and out.flags.writeable
        assert type(out.base).__name__ == "PyCapsule"  # the moved Grid, not a copy
    out = core.run_steps(init, [])  # no passes: a copy of init, not init itself
    assert np.array_equal(out, init) and out.ctypes.data != init.ctypes.data


def test_in_place_write_is_zero_copy(core):
    a = np.full((5, 7), -1, dtype=np.int64)
    assert core.data_address(a) == a.ctypes.data
    core.fill_flat_index(a)  # writes land in NumPy's buffer, not a copy
    assert np.array_equal(a, np.arange(35, dtype=np.int64).reshape(5, 7))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 6), dtype=np.int64)[:, ::2],  # non-contiguous view
        np.zeros((4, 6), dtype=np.int64, order="F"),  # Fortran order
        np.zeros((4, 6), dtype=np.int32),  # wrong dtype
    ],
    ids=["strided", "fortran", "int32"],
)
def test_in_place_write_refuses_arrays_that_would_be_copied(core, bad):
    with pytest.raises(TypeError):
        core.fill_flat_index(bad)


@pytest.mark.parametrize("shape", [(1, 1), (3, 1), (1, 9), (37, 53), (0, 4)])
def test_roundtrip_returns_equal_array_owning_cpp_buffer(core, shape):
    rng = np.random.default_rng(0)
    a = rng.integers(-1, 2**40, size=shape, dtype=np.int64)
    out = core.roundtrip(a)
    assert out.dtype == np.int64
    assert out.shape == a.shape
    assert out.flags.c_contiguous
    assert np.array_equal(out, a)
    if a.size:
        assert out.ctypes.data != a.ctypes.data  # a new buffer, owned by the Grid
    before = a.copy()
    out[...] = 7  # writable, and independent of the input
    assert np.array_equal(a, before)


def test_roundtrip_converts_non_contiguous_and_other_dtypes(core):
    base = np.arange(48, dtype=np.int32).reshape(6, 8)
    view = base[::2, 1::3]
    assert np.array_equal(core.roundtrip(view), view.astype(np.int64))


def test_roundtrip_rejects_wrong_ndim(core):
    with pytest.raises(ValueError, match="2D"):
        core.roundtrip(np.zeros(5, dtype=np.int64))
