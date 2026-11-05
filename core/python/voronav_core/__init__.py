"""voronav's optional C++ core. Install with `pip install -e ./core` from the
repository root; the main package selects it with `backend="cpp"`."""

from ._core import (
    brute_force,
    data_address,
    distance_sq,
    fill_flat_index,
    hardware_threads,
    init_nearest,
    roundtrip,
    run_steps,
    step_schedule,
    version,
)

__all__ = [
    "brute_force",
    "data_address",
    "distance_sq",
    "fill_flat_index",
    "hardware_threads",
    "init_nearest",
    "roundtrip",
    "run_steps",
    "step_schedule",
    "version",
]
__version__ = version()
