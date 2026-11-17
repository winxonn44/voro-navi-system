"""voronav's optional C++ core. Install with `pip install -e ./core` from the
repository root; the main package selects it with `backend="cpp"` (JFA) and
`build_roadmap(..., builder="cpp")` (the roadmap)."""

from ._core import (
    boundary_sites,
    brute_force,
    build_roadmap,
    connect_pieces,
    data_address,
    distance_sq,
    fill_flat_index,
    gvd_mask,
    hardware_threads,
    init_nearest,
    label_components,
    minimum_spanning_tree,
    obstacle_ids,
    reachable_components,
    resolve_kernel,
    roundtrip,
    run_steps,
    source_field,
    step_schedule,
    version,
)

__all__ = [
    "boundary_sites",
    "brute_force",
    "build_roadmap",
    "connect_pieces",
    "data_address",
    "distance_sq",
    "fill_flat_index",
    "gvd_mask",
    "hardware_threads",
    "init_nearest",
    "label_components",
    "minimum_spanning_tree",
    "obstacle_ids",
    "reachable_components",
    "resolve_kernel",
    "roundtrip",
    "run_steps",
    "source_field",
    "step_schedule",
    "version",
]
__version__ = version()
