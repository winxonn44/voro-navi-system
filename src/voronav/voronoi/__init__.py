"""Discrete Voronoi diagrams: JFA backends plus a brute-force oracle.

`voronoi(sites, ...)` is the convenience entry point (numpy in, numpy out).
`make_runner(...)` is for benchmarking: it does all host/device setup up
front and returns a callable that times only the JFA passes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .common import NO_SITE, distance_sq, init_nearest, step_schedule
from .oracle import brute_force

__all__ = [
    "NO_SITE",
    "brute_force",
    "distance_sq",
    "init_nearest",
    "make_runner",
    "step_schedule",
    "voronoi",
]


def make_runner(
    sites: np.ndarray,
    backend: str = "numpy",
    variant: str = "jfa+1",
    device: str = "cpu",
    threads: int | None = None,
    schedule: str = "dynamic",
) -> tuple[Callable[[], None], Callable[[], np.ndarray]]:
    """Return (run, fetch). `run()` executes JFA and blocks until it is done;
    `fetch()` returns the last result as a numpy array. `schedule` is
    numba-only ("dynamic" or "static" row distribution)."""
    init = init_nearest(sites)
    steps = step_schedule(*sites.shape, variant)
    result: dict[str, object] = {}

    if backend == "numpy":
        from . import jfa_numpy

        def run() -> None:
            result["out"] = jfa_numpy.run_steps(init, steps)

        def fetch() -> np.ndarray:
            return result["out"]  # type: ignore[return-value]

    elif backend == "numba":
        from . import jfa_numba

        def run() -> None:
            result["out"] = jfa_numba.run_steps(init, steps, threads, schedule)

        def fetch() -> np.ndarray:
            return result["out"]  # type: ignore[return-value]

    elif backend == "torch":
        from . import jfa_torch

        init_dev = jfa_torch.to_device(init, device)

        def run() -> None:
            result["out"] = jfa_torch.run_steps(init_dev, steps)
            jfa_torch.synchronize(device)

        def fetch() -> np.ndarray:
            return result["out"].cpu().numpy()  # type: ignore[union-attr]

    else:
        raise ValueError(f"unknown backend {backend!r}")
    return run, fetch


def voronoi(
    sites: np.ndarray,
    backend: str = "numpy",
    variant: str = "jfa+1",
    device: str = "cpu",
    threads: int | None = None,
    schedule: str = "dynamic",
) -> np.ndarray:
    """Nearest-site grid (flat site index per pixel, NO_SITE if no sites)."""
    run, fetch = make_runner(sites, backend, variant, device, threads, schedule)
    run()
    return fetch()
