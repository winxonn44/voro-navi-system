import pytest

from voronav.voronoi.jfa_torch import available_devices

# (backend, device) pairs every property test runs against. MPS/CUDA are
# included only when present, so the suite stays green on any laptop.
BACKENDS = [("numpy", "cpu"), ("numba", "cpu")] + [
    ("torch", d) for d in available_devices()
]


@pytest.fixture(params=BACKENDS, ids=lambda p: f"{p[0]}-{p[1]}")
def backend(request):
    return request.param
