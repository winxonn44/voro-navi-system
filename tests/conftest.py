import importlib.util

import pytest

from voronav.voronoi.jfa_torch import available_devices

# The optional C++ core (core/, `pip install -e ./core`). Without it the cpp
# rows are simply absent, so the suite stays green on a pure-Python install.
HAVE_CORE = importlib.util.find_spec("voronav_core") is not None

# (backend, device) pairs every property test runs against. MPS/CUDA and the
# C++ core are included only when present, so the suite stays green on any
# laptop.
BACKENDS = (
    [("numpy", "cpu"), ("numba", "cpu")]
    + [("torch", d) for d in available_devices()]
    + ([("cpp", "cpu")] if HAVE_CORE else [])
)


@pytest.fixture(params=BACKENDS, ids=lambda p: f"{p[0]}-{p[1]}")
def backend(request):
    return request.param
