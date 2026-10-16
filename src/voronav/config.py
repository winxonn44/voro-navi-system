"""Dataclass configs loaded from YAML.

Unknown keys raise instead of being ignored: a typo in a benchmark config
should fail before the sweep runs, not silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

BACKENDS = ("numpy", "numba", "torch")
VARIANTS = ("jfa", "jfa+1", "jfa+2", "1+jfa")
NUMBA_SCHEDULES = ("dynamic", "static")
MAP_KINDS = ("points", "obstacles", "corridor", "movingai", "image")


@dataclass
class MapConfig:
    kind: str = "points"
    height: int = 256
    width: int = 256
    n_sites: int = 64        # points: number of sites; obstacles: number of shapes
    seed: int = 0
    path: str | None = None  # movingai / image only

    def __post_init__(self) -> None:
        if self.kind not in MAP_KINDS:
            raise ValueError(f"map.kind must be one of {MAP_KINDS}, got {self.kind!r}")
        if self.kind in ("movingai", "image") and not self.path:
            raise ValueError(f"map.kind={self.kind!r} requires map.path")


@dataclass
class VoronoiConfig:
    backend: str = "numpy"
    variant: str = "jfa+1"
    device: str = "cpu"      # torch only: cpu | mps | cuda
    threads: int | None = None  # numba only; None = all cores

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"voronoi.backend must be one of {BACKENDS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"voronoi.variant must be one of {VARIANTS}")


@dataclass
class BenchConfig:
    """Grid sweep for the Q1 scaling study."""

    sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    site_densities: list[float] = field(default_factory=lambda: [1e-3])  # sites / pixel
    backends: list[str] = field(default_factory=lambda: ["numpy", "numba"])
    variants: list[str] = field(default_factory=lambda: ["jfa+1"])
    numba_threads: list[int] = field(default_factory=lambda: [1])
    numba_schedules: list[str] = field(default_factory=lambda: ["dynamic"])
    torch_devices: list[str] = field(default_factory=lambda: ["cpu"])
    repeats: int = 5
    warmup: int = 1
    measure_error: bool = True
    seed: int = 0
    out_dir: str = "runs/bench"
    # Pause after each timed configuration. On a fanless laptop (MacBook Air)
    # sustained all-core load throttles the chip, which slows later rows and
    # biases comparisons toward whatever ran first.
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        for b in self.backends:
            if b not in BACKENDS:
                raise ValueError(f"bench.backends: unknown backend {b!r}")
        for v in self.variants:
            if v not in VARIANTS:
                raise ValueError(f"bench.variants: unknown variant {v!r}")
        for sch in self.numba_schedules:
            if sch not in NUMBA_SCHEDULES:
                raise ValueError(f"bench.numba_schedules: unknown schedule {sch!r}")
        if self.repeats < 1:
            raise ValueError("bench.repeats must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("bench.cooldown_s must be >= 0")


Q2_PLANNERS = ("voronoi", "astar", "weighted_astar", "rrt_star")


@dataclass
class MapSourceConfig:
    """One entry of the Q2 map list; see voronav.map_sources for the kinds
    and which fields each one takes (unused fields are rejected)."""

    kind: str = "random_obstacles"
    sizes: list[int] | None = None   # synthetic: map side (height) in pixels
    seeds: list[int] | None = None   # random_obstacles only; default [0]
    n_shapes: int | None = None      # random_obstacles
    gap: int | None = None           # corridor: free width in pixels
    dir: str | None = None           # *_dir kinds
    max_maps: int | None = None      # *_dir kinds: first k files (sorted)

    def __post_init__(self) -> None:
        from .map_sources import validate_spec  # lazy: map_sources imports maps imports config

        validate_spec(self)


@dataclass
class RoadmapConfig:
    """How the Q2 harness builds each GVD roadmap (see build_roadmap)."""

    backend: str = "numba"
    variant: str = "jfa+2"
    min_angle_deg: float = 60.0
    raster_slack_px: float = 2.0
    min_piece_px: int = 8

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"roadmap.backend must be one of {BACKENDS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"roadmap.variant must be one of {VARIANTS}")


@dataclass
class Q2BenchConfig:
    """Planner comparison for the Q2 study: every planner runs on the same
    seeded start/goal queries of every map."""

    maps: list[MapSourceConfig] = field(
        default_factory=lambda: [MapSourceConfig("random_obstacles", sizes=[256], n_shapes=25)]
    )
    queries_per_map: int = 20
    # start and goal at least this fraction of the map diagonal apart
    min_dist_frac: float = 1 / 3
    seed: int = 0
    robot_radius: float = 2.0
    planners: list[str] = field(default_factory=lambda: list(Q2_PLANNERS))
    lambdas: list[float] = field(default_factory=lambda: [1.0, 4.0, 16.0, 64.0])
    rrt_iters: int = 2000
    rrt_every: int = 5      # RRT* (pure Python) runs on every k-th query only
    cooldown_s: float = 0.0  # pause after each map (fanless laptop, see BenchConfig)
    out_dir: str = "runs/bench_q2"

    def __post_init__(self) -> None:
        self.maps = [m if isinstance(m, MapSourceConfig) else _build(MapSourceConfig, m)
                     for m in self.maps]
        if not self.maps:
            raise ValueError("q2.maps must list at least one map source")
        for p in self.planners:
            if p not in Q2_PLANNERS:
                raise ValueError(f"q2.planners: unknown planner {p!r} (one of {Q2_PLANNERS})")
        if len(set(self.planners)) != len(self.planners):
            raise ValueError("q2.planners has duplicates")
        if "weighted_astar" in self.planners and not self.lambdas:
            raise ValueError("q2.lambdas must be non-empty when weighted_astar runs")
        if any(lam < 0 for lam in self.lambdas):
            raise ValueError("q2.lambdas must be >= 0")
        if self.queries_per_map < 1:
            raise ValueError("q2.queries_per_map must be >= 1")
        if not 0.0 <= self.min_dist_frac < 1.0:
            raise ValueError("q2.min_dist_frac must be in [0, 1)")
        if self.robot_radius < 0:
            raise ValueError("q2.robot_radius must be >= 0")
        if self.rrt_iters < 1 or self.rrt_every < 1:
            raise ValueError("q2.rrt_iters and q2.rrt_every must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("q2.cooldown_s must be >= 0")


@dataclass
class Q2Config:
    roadmap: RoadmapConfig = field(default_factory=RoadmapConfig)
    q2: Q2BenchConfig = field(default_factory=Q2BenchConfig)


@dataclass
class Config:
    map: MapConfig = field(default_factory=MapConfig)
    voronoi: VoronoiConfig = field(default_factory=VoronoiConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        sub = _NESTED.get((cls.__name__, name))
        kwargs[name] = _build(sub, value) if sub is not None else value
    return cls(**kwargs)


_NESTED = {
    ("Config", "map"): MapConfig,
    ("Config", "voronoi"): VoronoiConfig,
    ("Config", "bench"): BenchConfig,
    ("Q2Config", "roadmap"): RoadmapConfig,
    ("Q2Config", "q2"): Q2BenchConfig,
}


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)


def load_q2_config(path: str | Path) -> Q2Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Q2Config, raw)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj
