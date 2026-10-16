# Benchmark maps (Q2)

The Q2 planner study (Voronoi roadmap vs grid A* vs RRT*) runs on 70 real
maps: 50 MovingAI grid benchmarks in 5 sets of 10, plus 20 HouseExpo floor
plans. None of them are committed. `docs/maps_manifest.json` pins each one by
source URL and sha256, and `scripts/fetch_maps.py` fetches the maps and
verifies them against it.

## Fetching

```bash
.venv/bin/python -I scripts/fetch_maps.py --dry-run   # list archives and sizes
.venv/bin/python -I scripts/fetch_maps.py             # download, extract, verify
.venv/bin/python -I scripts/fetch_maps.py --no-scen   # skip MovingAI .scen files
```

The script downloads 11 archives, about 44 MB in total, into
`data/raw/archives/`. That directory is gitignored, and the archives are only
downloaded once. The script then extracts just the selected files:

```
data/raw/movingai/<set>/<map>.map        occupancy grid
data/raw/movingai/<set>/<map>.map.scen   official start/goal problems
data/raw/houseexpo/<id>.json             floor-plan contour
```

Every extracted file must match the sha256 in the manifest, and every map
must load at the pixel size the manifest records. If an upstream archive
changes, the script only warns, because the files inside are still checked
one by one. The script reads archive members by name and never calls
`extractall`, so a malformed archive cannot write outside `data/raw/`.

| Archive | Source | Size |
|---|---|---|
| `dao-map.zip`, `dao-scen.zip` | https://movingai.com/benchmarks/dao/ | 0.81 + 1.96 MB |
| `bg512-map.zip`, `bg512-scen.zip` | https://movingai.com/benchmarks/bg512/ | 0.30 + 1.63 MB |
| `street-map.zip`, `street-scen.zip` | https://movingai.com/benchmarks/street/ | 1.00 + 2.74 MB |
| `maze-map.zip`, `maze-scen.zip` | https://movingai.com/benchmarks/maze/ | 0.68 + 7.72 MB |
| `room-map.zip`, `room-scen.zip` | https://movingai.com/benchmarks/room/ | 0.21 + 1.09 MB |
| `houseexpo-json.tar.gz` | `HouseExpo/json.tar.gz` at commit `45e2b25` of https://github.com/TeaganLi/HouseExpo | 25.60 MB |

`--select` is for the maintainer only. It re-applies the selection rules
below to the downloaded archives and rewrites the manifest. Since the rules
are deterministic, rerunning it on the same archives produces the same
manifest.

## Loading

- MovingAI: `voronav.maps.load_movingai(path)`. `@ O T W` are obstacles and
  `. G S` are free.
- HouseExpo: `voronav.maps.load_houseexpo(path, px_per_meter=20, pad=2)`.
  `verts` is one closed contour of the house in meters. The inside is free
  floor; everything outside is obstacle, including the walls, because the
  contour wraps around them. A pixel is free when its center lies inside the
  contour (even-odd rule). Rows follow +y, as in the dataset's own PNG
  renderer, and a 2 px obstacle frame keeps the free space enclosed.

Both loaders return `bool[H, W]` with True = obstacle.

**HouseExpo resolution: 20 px/m (5 cm cells).** The dataset's viewer uses
100 px/m, which makes a typical 10–25 m house 1000–2500 px wide, too heavy for
a fanless laptop. At 20 px/m the same house is 200–500 px, inside the 512 px
budget of the other maps, while a 0.8 m door is still 16 px wide and a 10 cm
interior wall is still 2 px thick. A 5 cm cell is also a common resolution for
indoor occupancy grids. The selected houses range from 106x175 to 348x508 px.

## Why these maps

The study asks how planners behave on different kinds of free space, so each
set covers one structure. All maps are at most 512 px on a side, which keeps a
full Q2 run light on a fanless MacBook Air.

| Set | Structure | Selection rule (in `fetch_maps.py`) |
|---|---|---|
| `dao` (Dragon Age: Origins) | Irregular game levels: caves, dungeons, open areas | Maps with sides from 128 to 512 px and at least 20% free space, sorted by name; 10 picked at even spacing through the list |
| `bg512` (Baldur's Gate II, scaled to 512x512) | Outdoor and indoor levels with organic, blobby obstacles | Same rule as `dao` |
| `street` (city maps) | Building footprints with long, open streets | `<City>_0_512` for each of the 10 cities |
| `maze` | Corridors of one fixed width | Corridor widths 2, 4, 8, 16 and 32 px, instances 0 and 1. Width-1 mazes are left out because they have no clearance for the roadmap to maximize. |
| `room` | Grids of square rooms with random doorways | Room sizes 8, 16, 32 and 64 px; instances `000`–`001` for 8 and 16, `000`–`002` for 32 and 64 |
| `houseexpo` | Real indoor floor plans with walls, doors and corridors | Sorted ids shuffled with `numpy.random.default_rng(0)`; the first 20 houses with 4–12 rooms that fit in 512 px at 20 px/m |

The mazes and rooms are controlled cases where clearance is set by one known
parameter. `dao`, `bg512` and `street` show how planners behave on real
benchmark levels, and HouseExpo is closest to the indoor robot setting the
project targets. Each map also ships with its official MovingAI `.scen` file,
so results can be compared against published optimal path lengths as well as
the harness's own seeded queries.

## Licenses and attribution

**MovingAI.** The benchmarks page states that all data is released under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/)
(ODC-By). Please cite:

> N. Sturtevant. *Benchmarks for Grid-Based Pathfinding.* IEEE Transactions on
> Computational Intelligence and AI in Games, 4(2):144–148, 2012.

The Dragon Age and Baldur's Gate maps come from BioWare games and are
redistributed with BioWare's permission for research. The city/street maps
were provided by Konstantin Yakovlev and Anton Andreychuk. The maze and room
maps were generated by Nathan Sturtevant (HOG2).

**HouseExpo.** The repository is MIT-licensed (Copyright (c) 2019 Tingguang
Li). Please cite:

> T. Li, D. Ho, C. Li, D. Zhu, C. Wang, M. Q.-H. Meng. *HouseExpo: A
> Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms on Mobile
> Robots.* arXiv:1903.09845, 2019 (IROS 2020).

The floor plans are derived from the SUNCG dataset, which SUNCG's authors
released for non-commercial research and which is no longer publicly
distributed. This project uses the plans for academic research only and does
not redistribute them: only their hashes and sizes are committed.
