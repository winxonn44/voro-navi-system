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


def test_cpp_backend_is_not_implemented_yet(core):
    with pytest.raises(NotImplementedError, match="no JFA yet"):
        make_runner(random_sites(8, 8, 2, seed=0), backend="cpp")


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
