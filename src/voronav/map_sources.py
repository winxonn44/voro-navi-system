"""Map source registry for the Q2 planner benchmark.

A *source spec* (`MapSourceConfig`, from YAML) expands into one or more named
occupancy grids:

  synthetic, one map per (size, seed):
    random_obstacles  random rectangles and discs (`n_shapes`, `sizes`, `seeds`)
    corridor          straight corridor of width `gap`; deterministic
    two_rooms         two rooms joined by a door, with furniture; deterministic
  directory, one map per matching file (sorted, up to `max_maps`):
    movingai_dir      every *.map file in `dir`, via maps.load_movingai
    image_dir         every *.png file in `dir`, via maps.load_image_mask

Adding a directory format is one entry in `DIR_SOURCES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np

from . import maps as _maps

if TYPE_CHECKING:
    from .config import MapSourceConfig


@dataclass
class MapInstance:
    name: str            # unique within a run, e.g. "random_obstacles-256-s1"
    source: str          # the spec's kind
    occ: np.ndarray      # bool [H, W], True = obstacle
    params: dict = field(default_factory=dict)  # what generated it (size, seed, file)


def two_rooms(height: int = 120, width: int = 200) -> np.ndarray:
    """Two rooms split by a 4-px wall with one door, plus two pieces of
    furniture. At 120 x 200 this is exactly the map in scripts/plan_demo.py;
    other sizes scale the layout (wall and outer-frame thickness stay fixed)."""
    if height < 40 or width < 40:
        raise ValueError("two_rooms needs at least 40 x 40")
    occ = np.ones((height, width), dtype=bool)
    mid = width // 2
    occ[3:-3, 3 : mid - 2] = False
    occ[3:-3, mid + 2 : -3] = False
    door = max(3, round(height / 20))
    occ[height // 2 - door : height // 2 + door, mid - 2 : mid + 2] = False

    def box(y0, y1, x0, x1):  # fractions of the 120 x 200 layout
        occ[round(y0 * height / 120) : round(y1 * height / 120),
            round(x0 * width / 200) : round(x1 * width / 200)] = True

    box(20, 45, 30, 60)
    box(70, 100, 140, 150)
    return occ


# kind -> (generator(size, seed, spec) -> occ, uses seeds, required spec fields)
SYNTHETIC: dict[str, tuple[Callable, bool, tuple[str, ...]]] = {
    "random_obstacles": (
        lambda size, seed, spec: _maps.random_obstacles(size, size, spec.n_shapes, seed=seed),
        True, ("n_shapes",)),
    "corridor": (lambda size, seed, spec: _maps.corridor(size, size, spec.gap), False, ("gap",)),
    # 3:5 aspect like the demo map, so the door sits between two wide rooms
    "two_rooms": (lambda size, seed, spec: two_rooms(size, round(size * 5 / 3)), False, ()),
}

# kind -> (glob pattern, loader(path) -> occ)
DIR_SOURCES: dict[str, tuple[str, Callable[[Path], np.ndarray]]] = {
    "movingai_dir": ("*.map", _maps.load_movingai),
    "image_dir": ("*.png", _maps.load_image_mask),
    "houseexpo_dir": ("*.json", _maps.load_houseexpo),
}

KINDS = tuple(SYNTHETIC) + tuple(DIR_SOURCES)

_SYNTH_FIELDS = {"sizes", "seeds", "n_shapes", "gap"}
_DIR_FIELDS = {"dir", "max_maps"}


def validate_spec(spec: MapSourceConfig) -> None:
    """Raise ValueError when a spec names an unknown kind, misses a required
    parameter, or sets one its kind does not use (it would be ignored)."""
    if spec.kind not in KINDS:
        raise ValueError(f"map source kind must be one of {KINDS}, got {spec.kind!r}")
    set_fields = {
        name for name in _SYNTH_FIELDS | _DIR_FIELDS if getattr(spec, name) is not None
    }
    if spec.kind in SYNTHETIC:
        _, seeded, required = SYNTHETIC[spec.kind]
        allowed = {"sizes", *required} | ({"seeds"} if seeded else set())
        missing = {"sizes", *required} - set_fields
        if spec.sizes == [] or spec.seeds == []:
            raise ValueError(f"map source {spec.kind!r}: sizes / seeds must not be empty")
    else:
        allowed = _DIR_FIELDS
        missing = {"dir"} - set_fields
    if missing:
        raise ValueError(f"map source {spec.kind!r} requires {sorted(missing)}")
    extra = set_fields - allowed
    if extra:
        raise ValueError(f"map source {spec.kind!r} does not use {sorted(extra)}")
    if spec.max_maps is not None and spec.max_maps < 1:
        raise ValueError("map source max_maps must be >= 1")


def expand(spec: MapSourceConfig) -> Iterator[MapInstance]:
    """Generate or load every map a spec describes, lazily (big directories
    are read one file at a time)."""
    validate_spec(spec)
    if spec.kind in SYNTHETIC:
        gen, seeded, _ = SYNTHETIC[spec.kind]
        seeds = (spec.seeds or [0]) if seeded else [None]
        for size in spec.sizes:
            for seed in seeds:
                name = f"{spec.kind}-{size}" + (f"-s{seed}" if seeded else "")
                params = {"size": size} | ({"seed": seed} if seeded else {})
                yield MapInstance(name, spec.kind, gen(size, seed, spec), params)
        return
    pattern, loader = DIR_SOURCES[spec.kind]
    root = Path(spec.dir)
    if not root.is_dir():
        raise FileNotFoundError(f"map source {spec.kind!r}: no directory {root}")
    # recursive: fetch_maps.py stores MovingAI maps one folder per set
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"map source {spec.kind!r}: no {pattern} files in {root}")
    for path in files[: spec.max_maps]:
        rel = path.relative_to(root).with_suffix("")
        name = f"{spec.kind}-" + "-".join(rel.parts)  # e.g. movingai_dir-maze-maze512-1-0
        params = {"file": str(path)} | ({"set": rel.parts[0]} if len(rel.parts) > 1 else {})
        yield MapInstance(name, spec.kind, loader(path), params)


def expand_all(specs: list[MapSourceConfig]) -> Iterator[MapInstance]:
    seen: set[str] = set()
    for spec in specs:
        for m in expand(spec):
            if m.name in seen:
                raise ValueError(f"duplicate map name {m.name!r} (same source listed twice?)")
            seen.add(m.name)
            yield m
