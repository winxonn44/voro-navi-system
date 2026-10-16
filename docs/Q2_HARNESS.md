# Q2 harness: Voronoi roadmap vs grid A* vs clearance-weighted A* vs RRT*

Question: what does a GVD roadmap buy in clearance, what does it cost in
path length and time, and when does building it pay off? This page describes
the harness. The results go in the README once the full study (`bench_q2.yaml`)
has run plugged in.

```bash
python scripts/bench_plan.py --config configs/bench_q2_quick.yaml   # synthetic maps only, ~7 s
python scripts/plot_q2.py runs/bench_q2_quick/results.jsonl --out runs/bench_q2_quick/figs
python scripts/bench_plan.py --config configs/bench_q2.yaml         # full study, run plugged in
```

## Planners

| planner | what it does | query time includes |
|---|---|---|
| `voronoi` | Dijkstra from start and goal to the nearest skeleton pixel, then A* along the GVD skeleton | the three searches (roadmap build reported separately as `build_s`) |
| `astar` | 8-connected A*, no corner cutting, octile heuristic | the search |
| `weighted_astar` | same A*, but entering a pixel costs step length × (1 + λ / clearance). λ = 0 is plain A* | building the cost map (one NumPy pass) + the search |
| `rrt_star` | continuous-space RRT* with a fixed iteration budget | everything; pure Python, not comparable in wall time |

Clearance-weighted A* exists so Voronoi is not only compared with a planner
that hugs walls by construction. Every multiplier is ≥ 1, so the octile
heuristic stays admissible and the search stays optimal for the weighted
cost. `tests/test_grid_search.py` checks the weighted costs against SciPy's
csgraph Dijkstra on the same graph, and that unit weights reproduce the
unweighted search exactly. The unweighted code path is unchanged: a hash of
A* paths, costs, expansion counts, skeletons and Voronoi paths on four maps was
identical before and after the change.

Weighted A* reads the roadmap's JFA clearance field (`Roadmap.clearance`),
so like Voronoi it needs a distance transform per map. That cost is not in
its `plan_s`.

## Queries and metrics

* Start/goal pairs are seeded (`seed`, map index) and lie in the same
  8-connected region of the safe mask (free space with clearance ≥ the robot
  radius), at least `min_dist_frac` × map diagonal apart. If a map has no room
  for that many, the harness writes fewer and says so on stderr (`n_queries`
  in the map row).
* Every planner runs on the same queries. RRT* runs only on queries where
  `query % rrt_every == 0`; its rows carry the same `query` ids, so
  comparisons can be restricted to those queries.
* Path metrics (`length_px`, `min/mean/p10_clearance_px`, `total_turn_deg`,
  `collision_free`, `n_waypoints`) come from `path_metrics` with SciPy's
  exact EDT of the map, never the JFA field the roadmap was built from.

## Output (`results.jsonl`)

One line per map (`"row": "map"`): name, source and its parameters, size,
`free_frac`, `safe_frac`, `build_s`, `n_nodes`, `n_bridges`, `bridge_px`,
`n_pruned`, `n_queries`, the roadmap settings, and machine info.

One line per (query, planner, λ) (`"row": "query"`): map fields, `query`,
`start`, `goal`, `euclid_px`, `planner`, `lam` (null except weighted A*),
`success`, `plan_s`, `expansions`, `fail` (reason, if any), and every
`path_metrics` field (null on failure). Machine info is only on map rows to
keep the file small. `config.json` next to it stores the resolved config.

## Config

`configs/*q2*.yaml` load with `load_q2_config` into `Q2Config` (`roadmap:`
settings + `q2:` bench settings). Unknown keys raise, and so do map-source
fields that the source's kind does not use.

Map sources (`src/voronav/map_sources.py`):

| kind | fields | maps |
|---|---|---|
| `random_obstacles` | `sizes`, `seeds` (default `[0]`), `n_shapes` | one per size × seed |
| `corridor` | `sizes`, `gap` | one per size (deterministic) |
| `two_rooms` | `sizes` | one per size, height = size, width = 5/3 size; at 120 it is the `plan_demo.py` map |
| `movingai_dir` | `dir`, `max_maps` | every `*.map` in `dir` |
| `image_dir` | `dir`, `max_maps` | every `*.png` in `dir` |

A missing or empty directory raises, so a full run cannot silently skip
real maps. HouseExpo `*.json` maps need one line in `DIR_SOURCES` once
`maps.load_houseexpo` exists (it is there, commented out). `bench_q2.yaml`
expects MovingAI maps in `data/raw/movingai`. Change that to wherever
`scripts/fetch_maps.py` puts them.

## Figures (`plot_q2.py`)

* `q2_tradeoff.png`: mean and minimum clearance vs path length / A* length
  on the same query. The weighted-A* λ sweep is drawn as a curve.
* `q2_success.png`: path found and collision-free, per planner and λ.
* `q2_query_time.png`: query-time box plots, log scale.
* `q2_build_time.png`: roadmap build time vs map pixels, with a power-law fit.
* `q2_amortized.png`: Voronoi build + n × query vs n × A* query, per map
  (faint) and averaged, with the break-even n.

## First look (quick config, not the study)

6 synthetic maps (128² and 256² random obstacles, a corridor, two rooms),
16 queries each, robot radius 2 px. M4 MacBook Air, light load, one run:

| planner | solved | length / A* | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| Voronoi | 96/96 | 1.24 | 16.4 px | 4.9 px | 0.12 ms |
| A* | 96/96 | 1.00 | 10.2 px | 3.2 px | 0.13 ms |
| weighted A*, λ = 1 | 96/96 | 1.01 | 12.9 px | 4.0 px | 0.43 ms |
| weighted A*, λ = 4 | 96/96 | 1.03 | 14.1 px | 5.0 px | 0.61 ms |
| weighted A*, λ = 16 | 96/96 | 1.07 | 15.0 px | 5.2 px | 0.85 ms |
| weighted A*, λ = 64 | 96/96 | 1.11 | 16.0 px | 5.3 px | 0.94 ms |
| RRT* (1500 iters) | 23/24 | 0.96 | 10.6 px | 2.9 px | 74 ms |

The λ sweep traces a curve that ends close to Voronoi, and that is the
uncomfortable finding. At λ = 64, weighted A* gets 98% of Voronoi's mean
clearance with an 11% detour instead of 24%. Its *minimum* clearance is
higher than Voronoi's (5.3 vs 4.9 px). A likely cause, not yet checked: the
Voronoi entry and exit legs are shortest paths to the nearest skeleton pixel,
so they can graze corners. Voronoi's advantage is query time on a built
roadmap, about 6× faster than λ ≥ 4. Against plain A*, the mean roadmap build
(13.5 ms) pays off after about 200 queries per map. On maps this small, A* is
barely slower than a Voronoi query. RRT* paths are shorter than A* paths
because they move in continuous space, not on the 8-connected grid. All
successful paths from every planner were collision-free.
