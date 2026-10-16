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
    torch_devices: list[str] = field(default_factory=lambda: ["cpu"])
    repeats: int = 5
    warmup: int = 1
    measure_error: bool = True
    seed: int = 0
    out_dir: str = "runs/bench"

    def __post_init__(self) -> None:
        for b in self.backends:
            if b not in BACKENDS:
                raise ValueError(f"bench.backends: unknown backend {b!r}")
        for v in self.variants:
            if v not in VARIANTS:
                raise ValueError(f"bench.variants: unknown variant {v!r}")
        if self.repeats < 1:
            raise ValueError("bench.repeats must be >= 1")


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
}


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
