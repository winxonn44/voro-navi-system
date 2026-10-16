# Q2 harness: Voronoi roadmap vs grid A* vs clearance-weighted A* vs RRT*

Question: what does a GVD roadmap buy in clearance, what does it cost in
path length and time, and when does building it pay off? This page describes
the harness. The results go in the README once the full study (`bench_q2.yaml`)
has run plugged in.

```bash
python scripts/bench_plan.py --config configs/bench_q2_quick.yaml   # synthetic maps only, ~8 s
python scripts/plot_q2.py runs/bench_q2_quick/results.jsonl --out runs/bench_q2_quick/figs
python scripts/bench_plan.py --config configs/bench_q2.yaml         # full study, run plugged in
```

## Planners

| planner | what it does | query time includes |
|---|---|---|
| `voronoi` | get onto the GVD skeleton, A* along it, get off at the goal; how it joins and leaves, and the roadmap step cost, are per-variant options (below) | the searches and per-query cost maps (roadmap build reported separately as `build_s`) |
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

### Voronoi variants

`voronoi_plan(rm, start, goal, join, join_lam, roadmap_lam)`. The defaults
(`join="nearest"`, both λ = 0) are the original planner, bit-identical to
before (all 96 quick-config queries reproduce the earlier run's lengths,
clearances and expansion counts exactly).

* `join="nearest"`: Dijkstra from start and from goal to the first skeleton
  pixel reached, then A* along the skeleton between the two. The three
  choices are made separately, so the join pixel can be on a dead-end
  corner spur that the path then has to ride out (`test_best_join_removes_the_corner_spur_detour`).
* `join="best"`: one A* over three copies of the grid (entry leg → roadmap →
  exit leg, `grid_search.roadmap_search`) picks the join and leave pixels
  that minimize the total cost. A leg may not touch or diagonally cross the
  skeleton, so it stays in its endpoint's roadmap cell. Step costs use the
  same accounting as the separate searches, so the original planner's path
  is always a feasible answer and the joint cost is never higher (tested).
  The exit layer is restricted to the goal's cell (a flood fill), which made
  it 2 to 3× faster with identical answers.
* `join_lam`: legs cost step × (1 + λ / clearance). With `join="best"` it
  should be > 0: an unweighted leg costs the same per pixel as the roadmap,
  so the search could run the leg alongside the roadmap instead of joining it.
* `roadmap_lam`: roadmap steps cost step × (1 + λ / clearance), so the
  roadmap route prefers wide passages over a shorter squeeze.

The cost maps (`clearance_cost`) are built inside the timed query, as for
weighted A*.

### Bridges

`roadmap.bridge_lam` > 0 builds the repair bridges (`connect_pieces`) as
clearance-weighted paths instead of shortest ones. `Roadmap.bridges` is the
mask of pixels the bridges added; the map rows record `n_bridges` and
`bridge_px` as before. Default 0: bridges were never a path's bottleneck
(see below).

### Smoothing

`smooth_path(path, safe, clearance, mode)` (`planning/smoothing.py`) is a
post-process applied identically to every planner's path, RRT* included:
resample the path every 0.5 px, then greedy line-of-sight shortcutting from
each anchor to the farthest point one straight segment reaches. Segments are
checked every 0.25 px at the nearest pixel, the same rule as `segment_free`
and `path_metrics`. It reads the planner-side fields (`safe`, JFA clearance),
not the exact EDT the metrics use.

* `safe`: a segment must stay in `safe` (clearance ≥ robot radius).
* `preserve`: also, every sample's clearance must be ≥ the raw path's
  minimum between the segment's ends, so the minimum clearance never drops.

`tests/test_smoothing.py` checks with `path_metrics` that smoothed paths from
every planner are collision-free, never longer than the raw path, and (for
`preserve`) never lower in minimum clearance.

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
`variant` (the Voronoi variant's name, null for other planners), `success`,
`plan_s`, `expansions`, `fail` (reason, if any), and every `path_metrics`
field (null on failure). For every `q2.smoothing` mode the row also has the
same metric fields prefixed `<mode>_` (e.g. `preserve_min_clearance_px`) and
`<mode>_smooth_s`; `plan_s` never includes smoothing. Machine info is only on map rows to
keep the file small. `config.json` next to it stores the resolved config.

## Config

`configs/*q2*.yaml` load with `load_q2_config` into `Q2Config` (`roadmap:`
settings + `q2:` bench settings). Unknown keys raise, and so do map-source
fields that the source's kind does not use.

New in week 4b, all validated (unknown modes, negative λ, duplicate names
raise):

```yaml
roadmap:
  bridge_lam: 0            # > 0: clearance-weighted repair bridges
q2:
  voronoi_variants:        # one Voronoi row per entry; the first is the original
    - {name: nearest}
    - {name: nearest-road16, roadmap_lam: 16}
    - {name: best16, join: best, join_lam: 16, roadmap_lam: 16}
    - {name: best64, join: best, join_lam: 64, roadmap_lam: 64}
  smoothing: [safe, preserve]   # default []: raw metrics only
```

Without `voronoi_variants` the harness runs the original planner only, so
older configs keep their meaning.

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
  on the same query. The weighted-A* λ sweep is drawn as a curve; Voronoi
  variants share a color and differ in marker.
* `q2_tradeoff_<mode>.png`: the same after `<mode>` smoothing, with length
  relative to the smoothed A* path.
* `q2_success.png`: path found and collision-free, per planner and λ.
* `q2_query_time.png`: query-time box plots, log scale.
* `q2_build_time.png`: roadmap build time vs map pixels, with a power-law fit.
* `q2_amortized.png`: Voronoi build + n × query vs n × A* query, per map
  (faint) and averaged, with the break-even n. Uses the first Voronoi variant.

## First look (quick config, not the study): before week 4b

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
so they can graze corners. (Checked in week 4b, below: mostly wrong. The minimum was on
the roadmap itself, where the shortest route took a narrow gap.) Voronoi's advantage is query time on a built
roadmap, about 6× faster than λ ≥ 4. Against plain A*, the mean roadmap build
(13.5 ms) pays off after about 200 queries per map. On maps this small, A* is
barely slower than a Voronoi query. RRT* paths are shorter than A* paths
because they move in continuous space, not on the 8-connected grid. All
successful paths from every planner were collision-free.

## Week 4b: where Voronoi lost minimum clearance, and the fix

Same quick config and the same 96 queries, before any change. Each Voronoi
path was split into entry leg, roadmap (GVD pixels vs bridge pixels) and
exit leg, and graded with the exact EDT:

* In 56 of 96 queries the path minimum is the start or goal itself (29 / 27).
  No planner can beat that: every path starts and ends there.
* In the other 40 it is on the GVD part of the roadmap, never on a leg away
  from its endpoint and never on a bridge. Only 14 paths used a bridge at
  all, and their bridge minimum averaged 11.4 px against a path minimum of
  4.9 px. So the suspected causes were mostly wrong: entry/exit legs were
  not grazing corners, and bridges were not the problem.
* In 27 of those 40, a widest-path search on the skeleton between the same
  join and leave pixels found a route whose bottleneck was on average 2.35 px
  wider. The shortest roadmap route squeezes through a narrow gap (often a
  2 px one, the robot radius) when a wider route exists. That is a routing
  cost problem, not a join problem.

A ceiling for any planner: min(start clearance, goal clearance, widest-path
bottleneck in safe space between them) averages **5.42 px** over the 96
queries.

So two fixes, measured separately: clearance-weighted roadmap travel
(`roadmap_lam`), aimed at the measured cause, and the joint join/leave
(`join="best"`), aimed at detours such as corner spurs.

### Raw paths (96 queries, M4 MacBook Air, light load, one run)

| planner | solved | length / A* | mean clearance | min clearance | min at ceiling | median query |
|---|---|---|---|---|---|---|
| Voronoi, `nearest` (before) | 96/96 | 1.24 | 16.4 px | 4.93 px | 68/96 | 0.12 ms |
| Voronoi, `nearest-road16` | 96/96 | 1.26 | 17.6 px | 5.26 px | 81/96 | 0.18 ms |
| Voronoi, `best16` | 96/96 | 1.15 | 16.7 px | 5.24 px | 79/96 | 0.53 ms |
| Voronoi, `best64` | 96/96 | 1.17 | 17.1 px | 5.30 px | 84/96 | 0.52 ms |
| A* | 96/96 | 1.00 | 10.2 px | 3.16 px | | 0.13 ms |
| weighted A*, λ = 4 | 96/96 | 1.03 | 14.1 px | 5.01 px | | 0.58 ms |
| weighted A*, λ = 16 | 96/96 | 1.07 | 15.0 px | 5.24 px | 79/96 | 0.83 ms |
| weighted A*, λ = 64 | 96/96 | 1.11 | 16.0 px | 5.30 px | 84/96 | 0.97 ms |
| RRT* (1500 iters) | 23/24 | 0.96 | 10.6 px | 2.91 px | | 73 ms |

("min at ceiling": queries whose minimum clearance reaches the ceiling
above, from a separate script on the same queries.)

* The minimum-clearance gap is closed, but only up to a tie. `best64`
  matches weighted A* at λ = 64 exactly (5.30 px, the same 84 queries at the
  ceiling), with 7% more mean clearance and a 6-point longer detour (1.17 vs
  1.11).
* The cheapest fix is roadmap weighting alone (`nearest-road16`): +0.33 px
  minimum, +1.2 px mean, at 0.18 ms. It does not shorten anything.
* The joint join is what cuts the detour: 1.24 → 1.15 (`best16`). That is
  the corner-spur effect and its relatives, measured over all queries.
* It costs query time. Joint search is about 4× slower than the original
  (0.52 vs 0.12 ms median), so Voronoi's query-time lead over weighted A* at
  the same minimum clearance shrinks from about 8× to about 1.9× (λ = 64).
* Clearance-weighted bridges (`bridge_lam: 16`, separate run) moved a few
  bridge pixels (11 → 13 and 32 → 29 on two maps) and changed no path metric
  to two decimals. Left off.

### After smoothing (same paths; length relative to the smoothed A* path)

| planner | `preserve`: length / A* | mean | min | `safe`: length / A* | mean | min |
|---|---|---|---|---|---|---|
| Voronoi, `nearest` | 1.12 | 13.6 px | 4.94 px | 1.05 | 11.2 px | 2.91 px |
| Voronoi, `nearest-road16` | 1.13 | 14.5 px | 5.27 px | 1.05 | 11.6 px | 2.91 px |
| Voronoi, `best16` | 1.09 | 14.1 px | 5.24 px | 1.04 | 11.6 px | 2.94 px |
| Voronoi, `best64` | 1.11 | 14.4 px | 5.30 px | 1.05 | 11.7 px | 2.91 px |
| A* | 1.00 | 10.5 px | 3.17 px | 1.00 | 10.4 px | 2.91 px |
| weighted A*, λ = 16 | 1.06 | 13.2 px | 5.24 px | 1.03 | 11.4 px | 2.95 px |
| weighted A*, λ = 64 | 1.09 | 13.8 px | 5.30 px | 1.04 | 11.6 px | 2.92 px |
| RRT* | 1.00 | 10.6 px | 2.97 px | 1.00 | 10.5 px | 2.82 px |

Every smoothed path from every planner was collision-free. Smoothing takes
about 0.4 to 0.5 ms per path (median); RRT*'s sparse polylines take 0.15 ms.

* `safe` smoothing erases the differences. Every planner ends up within 5%
  of A*'s length, 10.4 to 11.7 px mean clearance and about 2.9 px minimum:
  shortcuts pull every path onto the same corners at the robot radius. Any
  clearance claim for a roadmap has to be made on raw or
  clearance-preserving paths.
* `preserve` keeps the ranking of the raw table and the minimum clearances
  (by construction). It shortens the original Voronoi paths most (1.24 →
  1.12), because those had the most detour to remove. After it, `best64` and
  weighted A* at λ = 64 have the same minimum, Voronoi has 0.5 px more mean
  clearance, and the length gap narrows to 2 points (1.11 vs 1.09).

Updated headline for the quick config: once the roadmap route is
clearance-weighted, Voronoi ties tuned weighted A* on minimum clearance,
keeps a small mean-clearance lead, and is still a little longer. Its query
time lead drops from about 8× to about 2× when the joint join is used. The
full study (`bench_q2.yaml`, which now carries the same four variants and
both smoothing modes) has to confirm this on real maps.
