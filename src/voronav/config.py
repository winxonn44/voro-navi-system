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
    bridge_lam: float = 0.0  # 0 = shortest bridges; > 0 = clearance-weighted

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"roadmap.backend must be one of {BACKENDS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"roadmap.variant must be one of {VARIANTS}")
        if self.bridge_lam < 0:
            raise ValueError("roadmap.bridge_lam must be >= 0")


JOIN_MODES = ("nearest", "best")
SMOOTH_MODES = ("safe", "preserve")


@dataclass
class VoronoiVariantConfig:
    """One setting of voronoi_plan; the defaults are the original planner
    (nearest skeleton pixel, shortest legs, shortest roadmap route)."""

    name: str = "nearest"
    join: str = "nearest"
    join_lam: float = 0.0
    roadmap_lam: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("q2.voronoi_variants: name must be non-empty")
        if self.join not in JOIN_MODES:
            raise ValueError(f"q2.voronoi_variants: join must be one of {JOIN_MODES}")
        if self.join_lam < 0 or self.roadmap_lam < 0:
            raise ValueError("q2.voronoi_variants: join_lam and roadmap_lam must be >= 0")


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
    # each runs as planner "voronoi" with its name in the rows' "variant"
    voronoi_variants: list[VoronoiVariantConfig] = field(
        default_factory=lambda: [VoronoiVariantConfig()]
    )
    # post-processes applied to every planner's path; raw metrics are kept
    smoothing: list[str] = field(default_factory=list)

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
        self.voronoi_variants = [v if isinstance(v, VoronoiVariantConfig)
                                 else _build(VoronoiVariantConfig, v)
                                 for v in self.voronoi_variants]
        if "voronoi" in self.planners and not self.voronoi_variants:
            raise ValueError("q2.voronoi_variants must be non-empty when voronoi runs")
        names = [v.name for v in self.voronoi_variants]
        if len(set(names)) != len(names):
            raise ValueError("q2.voronoi_variants: names must be unique")
        for mode in self.smoothing:
            if mode not in SMOOTH_MODES:
                raise ValueError(f"q2.smoothing: unknown mode {mode!r} (one of {SMOOTH_MODES})")
        if len(set(self.smoothing)) != len(self.smoothing):
            raise ValueError("q2.smoothing has duplicates")


Q3_PLANNERS = ("voronoi", "astar", "weighted_astar")
# the keys of voronav.corruption.CORRUPTIONS (tests check they agree)
CORRUPTION_KINDS = ("boundary_shift", "region_dropout", "thin_obstacle_miss", "boundary_noise")


def _check_corruption(where: str, spec: Any) -> tuple[str, float] | None:
    """A (registry name, level) pair, or None for no corruption."""
    if spec is None:
        return None
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError(f"{where} must be [name, level] or null, got {spec!r}")
    name, level = spec
    if name not in CORRUPTION_KINDS:
        raise ValueError(f"{where}: unknown corruption {name!r} (one of {CORRUPTION_KINDS})")
    return name, float(level)


@dataclass
class FramesConfig:
    """Q3 Part B: a stream of frames where each one is, with probability
    p_bad, a badly segmented frame; plan on every frame, raw and after
    temporal_filter. Skipped when `maps` is empty."""

    maps: list[MapSourceConfig] = field(default_factory=list)
    queries_per_map: int = 4
    n_frames: int = 20
    p_bad: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4])
    base: Any = field(default_factory=lambda: ["boundary_noise", 0.5])  # every good frame
    # each [name, level] is its own set of sequences (row field "bad")
    bad: list[Any] = field(default_factory=lambda: [["region_dropout", 0.4]])
    window: int = 5         # temporal_filter arguments
    outlier_iou: float = 0.8
    patience: int | None = 3

    def __post_init__(self) -> None:
        self.maps = [m if isinstance(m, MapSourceConfig) else _build(MapSourceConfig, m)
                     for m in self.maps]
        if self.queries_per_map < 1 or self.n_frames < 1:
            raise ValueError("q3.frames: queries_per_map and n_frames must be >= 1")
        if not self.p_bad or any(not 0.0 <= p <= 1.0 for p in self.p_bad):
            raise ValueError("q3.frames.p_bad must be non-empty, each in [0, 1]")
        self.base = _check_corruption("q3.frames.base", self.base)
        if not self.bad:
            raise ValueError("q3.frames.bad must list at least one [name, level]")
        self.bad = [_check_corruption("q3.frames.bad", b) for b in self.bad]
        if None in self.bad:
            raise ValueError("q3.frames.bad entries must be [name, level], not null")
        if self.window < 1 or not 0.0 <= self.outlier_iou <= 1.0:
            raise ValueError("q3.frames: window must be >= 1, outlier_iou in [0, 1]")
        if self.patience is not None and self.patience < 1:
            raise ValueError("q3.frames.patience must be >= 1 or null")


@dataclass
class Q3BenchConfig:
    """Q3: corrupt the true map, plan on the corrupted map, grade the path on
    the true one. Every planner runs on the same seeded queries, sampled in
    the true safe space."""

    maps: list[MapSourceConfig] = field(
        default_factory=lambda: [MapSourceConfig("cluttered_room", sizes=[256], seeds=[0])]
    )
    queries_per_map: int = 10
    min_dist_frac: float = 1 / 3
    seed: int = 0
    robot_radius: float = 2.0
    planners: list[str] = field(default_factory=lambda: list(Q3_PLANNERS))
    lambdas: list[float] = field(default_factory=lambda: [16.0, 64.0])
    voronoi_variants: list[VoronoiVariantConfig] = field(
        default_factory=lambda: [VoronoiVariantConfig()]
    )
    # corruption name -> levels; include the level that means "no error" as
    # the control (0 for every kind in voronav.corruption)
    corruptions: dict[str, list[float]] = field(
        default_factory=lambda: {"boundary_shift": [0.0, 2.0]}
    )
    draws: int = 1           # independent corruptions per (map, kind, level)
    cooldown_s: float = 0.0  # pause after each map (fanless laptop, see BenchConfig)
    out_dir: str = "runs/bench_q3"
    frames: FramesConfig = field(default_factory=FramesConfig)

    def __post_init__(self) -> None:
        self.maps = [m if isinstance(m, MapSourceConfig) else _build(MapSourceConfig, m)
                     for m in self.maps]
        if not self.maps:
            raise ValueError("q3.maps must list at least one map source")
        for p in self.planners:
            if p not in Q3_PLANNERS:
                raise ValueError(f"q3.planners: unknown planner {p!r} (one of {Q3_PLANNERS})")
        if not self.planners or len(set(self.planners)) != len(self.planners):
            raise ValueError("q3.planners must be non-empty without duplicates")
        if "weighted_astar" in self.planners and not self.lambdas:
            raise ValueError("q3.lambdas must be non-empty when weighted_astar runs")
        if any(lam < 0 for lam in self.lambdas):
            raise ValueError("q3.lambdas must be >= 0")
        self.voronoi_variants = [v if isinstance(v, VoronoiVariantConfig)
                                 else _build(VoronoiVariantConfig, v)
                                 for v in self.voronoi_variants]
        if "voronoi" in self.planners and not self.voronoi_variants:
            raise ValueError("q3.voronoi_variants must be non-empty when voronoi runs")
        names = [v.name for v in self.voronoi_variants]
        if len(set(names)) != len(names):
            raise ValueError("q3.voronoi_variants: names must be unique")
        if self.queries_per_map < 1:
            raise ValueError("q3.queries_per_map must be >= 1")
        if not 0.0 <= self.min_dist_frac < 1.0:
            raise ValueError("q3.min_dist_frac must be in [0, 1)")
        if self.robot_radius < 0:
            raise ValueError("q3.robot_radius must be >= 0")
        if not isinstance(self.corruptions, dict) or not self.corruptions:
            raise ValueError("q3.corruptions must map at least one corruption to its levels")
        for name, levels in self.corruptions.items():
            if name not in CORRUPTION_KINDS:
                raise ValueError(f"q3.corruptions: unknown corruption {name!r} "
                                 f"(one of {CORRUPTION_KINDS})")
            if not isinstance(levels, list) or not levels:
                raise ValueError(f"q3.corruptions.{name} must be a non-empty list of levels")
            if len(set(levels)) != len(levels):
                raise ValueError(f"q3.corruptions.{name} has duplicate levels")
        if self.draws < 1:
            raise ValueError("q3.draws must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("q3.cooldown_s must be >= 0")


@dataclass
class Q3Config:
    roadmap: RoadmapConfig = field(default_factory=RoadmapConfig)
    q3: Q3BenchConfig = field(default_factory=Q3BenchConfig)


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
    ("Q3Config", "roadmap"): RoadmapConfig,
    ("Q3Config", "q3"): Q3BenchConfig,
    ("Q3BenchConfig", "frames"): FramesConfig,
}


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)


def load_q2_config(path: str | Path) -> Q2Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Q2Config, raw)


def load_q3_config(path: str | Path) -> Q3Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Q3Config, raw)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj
