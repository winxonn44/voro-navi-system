"""Map source registry for the Q2 and Q3 planner benchmarks.

A *source spec* (`MapSourceConfig`, from YAML) expands into one or more named
occupancy grids:

  synthetic, one map per (size, seed):
    random_obstacles  random rectangles and discs (`n_shapes`, `sizes`, `seeds`)
    corridor          straight corridor of width `gap`; deterministic
    two_rooms         two rooms joined by a door, with furniture; deterministic
    cluttered_room    four rooms with thick walls and doors, cluttered with
                      thin obstacles (chair legs, poles, railings; `seeds`)
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


def cluttered_room(size: int = 256, seed: int = 0) -> np.ndarray:
    """A 2 x 2 grid of rooms (walls and frame >= 6 px thick, one door in each
    inner wall) cluttered with obstacles 1-4 px thin: table and chair legs
    (1-2 px dots), poles (2-4 px squares) and railings (1-2 px lines, some
    with posts). Built for Q3, where segmentation can erase thin obstacles
    (`thin_obstacle_miss`): walls survive that, the clutter is what vanishes.
    Clutter can stand against walls but keeps clear of the doors."""
    if size < 96:
        raise ValueError("cluttered_room needs size >= 96")
    rng = np.random.default_rng(seed)
    h = w = size
    wall = max(6, size // 40)
    occ = np.zeros((h, w), dtype=bool)
    occ[:wall], occ[-wall:], occ[:, :wall], occ[:, -wall:] = True, True, True, True
    my, mx = h // 2, w // 2
    occ[my - wall // 2 : my - wall // 2 + wall] = True
    occ[:, mx - wall // 2 : mx - wall // 2 + wall] = True
    door = max(10, size // 12)
    keep_free = np.zeros_like(occ)  # doors plus the approach to them
    for lo, hi in ((wall, mx - wall // 2), (mx - wall // 2 + wall, w - wall)):  # horizontal wall
        c = int(rng.integers(lo + door, hi - 2 * door))
        occ[my - wall // 2 : my - wall // 2 + wall, c : c + door] = False
        keep_free[my - wall - door : my + wall + door, c - door // 2 : c + door + door // 2] = True
    for lo, hi in ((wall, my - wall // 2), (my - wall // 2 + wall, h - wall)):  # vertical wall
        c = int(rng.integers(lo + door, hi - 2 * door))
        occ[c : c + door, mx - wall // 2 : mx - wall // 2 + wall] = False
        keep_free[c - door // 2 : c + door + door // 2, mx - wall - door : mx + wall + door] = True
    clutter = np.zeros_like(occ)
    margin = wall + 1  # clutter may stand against the walls, like real furniture

    def dot(y, x, s):
        clutter[y : y + s, x : x + s] = True

    rooms = [(y0, y1, x0, x1) for y0, y1 in ((0, my), (my, h)) for x0, x1 in ((0, mx), (mx, w))]
    scale = size / 256
    for ri, (y0, y1, x0, x1) in enumerate(rooms):
        ry0, ry1, rx0, rx1 = y0 + margin, y1 - margin, x0 + margin, x1 - margin

        def at(sh, sw):  # random top-left corner for an item of extent sh x sw
            return (int(rng.integers(ry0, max(ry0 + 1, ry1 - sh))),
                    int(rng.integers(rx0, max(rx0 + 1, rx1 - sw))))

        for _ in range(max(1, round(2 * scale))):  # table: 4 legs at the corners
            th, tw = int(rng.integers(12, 22)), int(rng.integers(16, 30))
            y, x = at(th, tw)
            s = int(rng.integers(1, 3))
            for dy, dx in ((0, 0), (0, tw - s), (th - s, 0), (th - s, tw - s)):
                dot(y + dy, x + dx, s)
        for _ in range(max(2, round(5 * scale))):  # chairs: 4 legs, 6-9 px apart
            c = int(rng.integers(6, 10))
            y, x = at(c, c)
            for dy, dx in ((0, 0), (0, c - 1), (c - 1, 0), (c - 1, c - 1)):
                dot(y + dy, x + dx, 1)
        for _ in range(max(2, round(4 * scale))):  # poles
            s = int(rng.integers(2, 5))
            dot(*at(s, s), s)
        if ri % 2 == 0:  # railing across part of the room, posts every 8 px on some
            t = int(rng.integers(1, 3))
            length = int(rng.integers((rx1 - rx0) * 2 // 5, (rx1 - rx0) * 3 // 5))
            if rng.random() < 0.5:
                y, x = at(t, length)
                clutter[y : y + t, x : x + length] = True
                if rng.random() < 0.5:
                    for px in range(x, x + length, 8):
                        dot(y - 1, px, t + 2)
            else:
                y, x = at(length, t)
                clutter[y : y + length, x : x + t] = True
    return occ | (clutter & ~keep_free)


# kind -> (generator(size, seed, spec) -> occ, uses seeds, required spec fields)
SYNTHETIC: dict[str, tuple[Callable, bool, tuple[str, ...]]] = {
    "random_obstacles": (
        lambda size, seed, spec: _maps.random_obstacles(size, size, spec.n_shapes, seed=seed),
        True, ("n_shapes",)),
    "corridor": (lambda size, seed, spec: _maps.corridor(size, size, spec.gap), False, ("gap",)),
    # 3:5 aspect like the demo map, so the door sits between two wide rooms
    "two_rooms": (lambda size, seed, spec: two_rooms(size, round(size * 5 / 3)), False, ()),
    "cluttered_room": (lambda size, seed, spec: cluttered_room(size, seed), True, ()),
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
