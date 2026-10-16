# voronav

From-scratch parallel Voronoi roadmaps for **offline, segmentation-driven
navigation**: SAM free-space masks → generalized Voronoi diagram →
maximum-clearance paths, running entirely on a laptop with no network.

The core is built without a geometry library: discrete Voronoi diagrams come
from my own Jump Flooding Algorithm (JFA) in three backends — a readable NumPy
reference, a multithreaded Numba CPU kernel and a PyTorch GPU kernel (Apple MPS
or CUDA) — all checked against a brute-force oracle. Every design decision is
visible and tested.

## Claim (and what it is not)

The project is designed to support one measurable claim:

> A maximum-clearance roadmap built by a from-scratch parallel Voronoi engine
> on top of SAM segmentation produces safer paths than grid A* at comparable
> planning time, and runs fully offline on a laptop.

It does **not** claim to be a full SLAM or autonomy stack. Localization is
studied only as the high-dimensional Voronoi chapter (Q5).

## Research questions

| # | Question | Status |
| --- | --- | --- |
| Q1 | How does parallel JFA scale across backends, threads and grid size, and how much error does each variant leave? | **results in** (below); GPU and scheduling causes measured and fixed |
| Q2 | Voronoi roadmap vs grid A* vs RRT*: clearance, length, time, success | **answered** (below): safer than A* in 99% of queries; ties tuned weighted A* on clearance, wins on query speed |
| Q3 | How does segmentation error carry through to path safety? | planned (week 5) |
| Q4 | SAM variants on-device: latency vs mask quality vs path quality | planned (week 6) |
| Q5 | High-dimensional Voronoi (k-means IVF) for place recognition: recall vs dimension | stretch (week 7) |

## Q1 first results (measured, Apple M4, 10 cores)

`configs/bench_q1.yaml` as committed: 4 grid sizes × 3 site densities × 4
variants × 15 backend configs (NumPy; Numba at 1–10 threads with static and
dynamic row scheduling; PyTorch CPU and MPS) = 720 rows, median of 5 runs
after 2 warmups. Raw data: `docs/runs/q1_results.jsonl`. The first run, from
before the fixes below, is kept in `docs/runs/q1_results_v1.jsonl`.

![JFA time per diagram](docs/q1/q1_throughput.png)

| `jfa+1`, 2048², 0.001 sites/px | first run | now | vs NumPy (now) |
| --- | --- | --- | --- |
| NumPy reference | 4165 ms | 4144 ms | 1× |
| PyTorch CPU (eager int64) | 2848 ms | 3140 ms | 1.3× |
| Numba, 1 thread | 334 ms | 330 ms | 13× |
| Numba, 10 threads, dynamic rows | (8 threads static: 90 ms) | 72 ms | 58× |
| **PyTorch MPS (fused Metal kernel)** | 1478 ms (eager int64) | **15 ms** | **276×** |

All of these produce the same output, bit for bit (`tests/test_backends_agree.py`).

**The GPU was slow because of unfused ops plus emulated 64-bit division.
Allocation was not the problem.** Both hypotheses from the first run were
measured one at a time (`scripts/probe_mps.py`, output in
`docs/runs/q1_mps_probe.json`, 2048², `jfa+1`):

| implementation | time | what changed |
| --- | --- | --- |
| eager PyTorch, int64 (old backend) | 1441 ms | 141 aten ops per pass, each a full-grid kernel |
| eager PyTorch, int32 | 825 ms | same ops, half the bytes |
| `torch.compile` of the int64 pass | 142 ms | ops fused by Inductor, still int64 math |
| fused Metal kernel, int64 storage + int64 math | 158 ms | one kernel per pass, packed int64 key |
| … int64 storage, a single int64 division per pixel | 45 ms | coordinates split with int32 division |
| … int64 storage, int32 math | 16 ms | no 64-bit arithmetic at all |
| **fused Metal kernel, int32 (new backend)** | **14 ms** | int32 storage and math |

- *Allocation (H2): rejected as stated.* Writing into a preallocated `out=`
  tensor is no faster than allocating (int32 add: 0.85 vs 0.90 ms). PyTorch's
  caching allocator makes allocation cheap. The real cost is what those
  temporaries imply: about 141 separate kernels per pass, each streaming whole
  grids through memory. Every eager op takes about the same time whatever it
  computes (add, floor-divide and remainder are all about 1.4 ms in int64), so
  the eager path is bandwidth-bound. Fusing the work into one kernel gives about 9×.
- *int64 (H1): confirmed, but only for division.* In eager mode int64 costs
  exactly its bytes: 1.7× over int32, with no sign of emulation. Inside a fused
  kernel, where memory traffic no longer dominates, int64 storage costs only
  10% (16 vs 14 ms). But 64-bit integer division and modulo are emulated on
  the Apple GPU: one per pixel triples the pass time, and the ten in the
  packed-key kernel make it 11× slower. `torch.compile` lands at the same
  ~145 ms for the same reason.
- *The fix:* one Metal kernel per pass (`torch.mps.compile_shader`) that
  keeps site indices and distances in int32. It compares `(d2, site)`
  lexicographically instead of building the packed int64 key, which orders
  candidates exactly the same way. Output is converted back to int64, so the
  API is unchanged. Grids over 2³¹ pixels or 46340 px per side fall back to
  the eager int64 path (`jfa_torch.fused_supported`). At 256² and 512² the
  GPU sits at a ~1.2 ms floor of launch and sync overhead, which is still
  slightly faster than Numba at 256² (1.3 ms).

![Numba strong scaling](docs/q1/q1_scaling.png)

**The 6-thread dip was load imbalance from static row splitting. Dynamic
scheduling removes it.** With `prange`'s static split, speedup at 2048² rises
to 3.3× at 4 threads, falls to 3.1× at 6, and ends at 3.6× at 10. The M4 has 4
performance and 6 efficiency cores, and each thread gets the same number of
rows, so the efficiency-core threads finish last. The fix that looks obvious
does nothing: `numba.set_parallel_chunksize` still gave every thread exactly
H / threads rows on both the workqueue and OpenMP layers, and timings did not
change. (A compute-only kernel with no memory traffic shows no dip, so the
efficiency cores fall further behind on JFA's memory-heavy passes.) The Numba
backend now claims rows 4 at a time from a shared atomic counter (an LLVM
`atomicrmw` through `numba.extending.intrinsic`), so faster cores take more
rows. Speedup is now monotonic: 3.4× at 4, 3.8× at 6, 4.1× at 8, 4.6× at 10
(72 ms). Chunks of 1, 4 and 16 rows measured within noise. The Amdahl fit
(p = 0.875) describes the dynamic curve reasonably; what remains is the
efficiency cores being slower, not a serial fraction.

![JFA error by variant](docs/q1/q1_error.png)

**Error:** plain JFA gets the distance wrong at 1.6e-4 of pixels at the highest
density, with the worst pixel off by 3.4 px. One extra k=1 pass helps more when
it runs *first*: `1+jfa` leaves 1.2e-6 versus 6.7e-6 for `jfa+1` at the same
cost. `jfa+2` was exact on every instance measured here (one seed per
configuration). The roadmap default should be `1+jfa` or `jfa+2`, with the
choice made in Q2 by whether the residual error changes paths.

Caveats: one seed, one machine, and laptop timings are sensitive to thermal
throttling and background load.

## Q2 results: Voronoi roadmap vs A*, weighted A* and RRT*

`configs/bench_q2.yaml` as committed: **94 maps** (20 random-obstacle maps at
128²–1024², a corridor, 3 two-room maps, 50 MovingAI maps in 5 sets, 20
HouseExpo houses), up to 50 seeded start/goal pairs per map, robot radius 2 px.
Every planner ran the same **4,100 queries**: 58,220 planner runs in 17
minutes on the M4 MacBook Air. All successful paths from every planner passed
the independent collision check (exact EDT), with **zero failures**. Grid A*,
weighted A* and every Voronoi variant solved 4,100 of 4,100; RRT* (3,000
iterations, every 5th query) solved 656 of 820. Raw data:
`docs/runs/q2_results.jsonl.gz` (`gunzip -k` to use); all tables: [`docs/q2/q2_summary.md`](docs/q2/q2_summary.md).

![Clearance vs detour](docs/q2/q2_tradeoff.png)

| planner (raw paths) | length / A* (median) | mean clearance | min clearance | median query |
| --- | --- | --- | --- | --- |
| Grid A* | 1.00 | 16.6 px | 2.93 px | 1.08 ms |
| Weighted A*, λ = 16 | 1.07 | 24.6 px | 6.98 px | 4.26 ms |
| Weighted A*, λ = 64 | 1.11 | 26.5 px | 7.29 px | 5.18 ms |
| **Voronoi, `nearest`** (original) | 1.25 | 28.1 px | 6.60 px | **0.66 ms** |
| Voronoi, `nearest-road16` | 1.27 | 28.8 px | 6.92 px | 0.89 ms |
| **Voronoi, `best64`** (joint join, weighted roadmap) | 1.18 | 28.2 px | 7.19 px | 3.63 ms |
| RRT* (pure Python, 656 solved) | 0.96 | 20.8 px | 3.31 px | 184 ms |

**Against grid A\*, the claim holds.** On the same queries, the original
Voronoi planner's minimum clearance was at least A*'s in **99%** of queries,
mean clearance was 11.5 px higher (+69%), and paths were 25% longer. Queries
were 1.6× faster once the roadmap was built. The build pays for itself after
about 120 queries per map:

![Amortized time](docs/q2/q2_amortized.png)

**Against a tuned clearance-weighted A\*, clearance is a tie, and speed is
the difference.** Weighted A* traces a smooth curve of length against
clearance, and the Voronoi variants sit at its high-clearance end. Paired by
query, `best64` against λ = 64:

- minimum clearance equal or higher in **95%** of queries (median difference
  exactly 0), mean clearance +1.7 px, paths 5.7% longer, queries 1.7× faster;
- the speedup grows with map size (2.4–2.7× on random 512²–1024²) and
  *reverses* on small maps (0.7× on HouseExpo houses, which average 311 px),
  where the three-layer joint join costs more than it saves;
- `nearest-road16` against λ = 16 trades the other way: 3.8× faster and
  +4.1 px mean clearance, but 16.5% longer.

So the honest headline is: **Voronoi roadmaps match the safety of a tuned
weighted A\*, and are faster per query on large maps once built.** It is not
that they are uniquely safe.

**Where Voronoi loses:**

1. **Rare but large minimum-clearance losses.** In 72 of 4,100 queries, `best64`'s
   narrowest point was more than 2 px tighter than weighted A*'s (worst
   16 px), concentrated on street maps, HouseExpo and Baldur's Gate. That's
   why its *average* minimum (7.19 px) trails λ = 64 (7.29 px) despite the
   95% tie-or-win rate. The cause has not been diagnosed yet.
2. **Narrow-passage maps** (supplement, below): 1.8–2.4× longer paths on room
   maps for about 1 px more clearance.
3. **Build cost on fragmented skeletons:** Baldur's Gate maps average 62
   bridges and 220 ms to build, against 10–100 ms elsewhere at similar size.

**Smoothing** (applied identically to every planner, never lowering the
minimum clearance) closes most of the length gap. Smoothed `best64` is 1.13×
A* length against 1.10× for smoothed λ = 64 weighted A*, with clearances
unchanged.

### Supplement: narrow-passage maps (robot radius 1 px)

At radius 2 px, all 10 MovingAI *room* maps and 2 of the *maze* maps
produced **no queries**: their 1–2 px doors and corridors are closed to a
radius-2 robot, so no start/goal pair lies in one reachable region. The
harness logged this, but the first summary silently dropped those maps, so
they are reported here as a separate condition (`configs/bench_q2_narrow.yaml`;
20 maps, 1,000 queries, 14,200 runs, zero collisions; raw data
`docs/runs/q2_narrow_results.jsonl.gz`;
[`docs/q2/q2_narrow_summary.md`](docs/q2/q2_narrow_summary.md)):

| set | Voronoi `nearest` vs A*: length | mean clearance gain | `best64` vs λ = 64 weighted A*: length | speedup |
| --- | --- | --- | --- | --- |
| maze (10) | 1.24× | +3.6 px | 1.02× | 1.9× |
| room (10) | **2.02×** | +3.2 px | **1.89×** | 4.1× |

Room maps are Voronoi's worst case. Every 1 px door limits all planners to
the same minimum clearance, so the skeleton's detours through room centers
buy almost nothing. The repair is also expensive here: the 8-room maps need
about 2,950 bridges and 25 s to build, because `connect_pieces` runs one
search per bridge. RRT* solved 1 of 200 queries in these maps, the known
failure of sampling planners in narrow passages.

### Caught by a fail-loud check

The first full run crashed on MovingAI `dao/lak303d` with "skeleton piece
unreachable inside its own region", an assertion in the roadmap repair. Free
regions had been labeled with 8-connectivity, but searches may not cut
corners, so two areas touching only at a corner are not mutually reachable:
that map has 4 regions under 8-connectivity but 24 reachable ones. Regions
and skeleton pieces are now labeled by `reachable_components`, using the
search's exact move rule, and a regression test covers the diagonal pinch.
The query sampler had the same mismatch and was fixed too. The run was
restarted from scratch, so no reported number mixes old and new code.

## Week 3: roadmap and planners

![Planners on example maps](docs/q2/q2_example.png)

- **Roadmap:** obstacle boundary pixels → JFA (`jfa+2`, exact in every Q1
  instance) → generalized Voronoi diagram (GVD) pixels → one connected skeleton
  per free region. It takes 3–20 ms at 256² on the M4.
- **Planners:** the Voronoi planner joins the skeleton at the nearest skeleton
  pixel, follows it with A*, and leaves it at the goal. Grid A* and
  multi-source Dijkstra are compiled with Numba and share one search routine.
  RRT* is a pure-Python baseline, so only its path quality is comparable, not
  its timing.
- **Metrics** use SciPy's exact distance transform, independent of the
  JFA field the roadmap was built on, so the planner doesn't grade its own work.
- **First look:** on 15 random queries over 4 maps (robot radius 2 px),
  Voronoi paths averaged 21.6 px mean clearance against 13.9 for A*, and were
  about 24% longer. The full comparison is in Q2 results, above.

Three design problems were caught while building this, and all are kept as tests:

1. **The angle rule broke real roadmap edges.** The first test for a GVD edge
   was angle-only: the two nearest sites must subtend at least 60°. It removed
   real boundaries where two different obstacles lie in similar directions
   from a pixel, splitting one free region's roadmap into about 120 pieces.
   Now sites on *different* obstacles always form an edge, and the angle rule
   (with a 2 px tolerance for pixel staircases on curves) applies only within
   one obstacle. Obstacles are labeled with the outer 1 px frame removed, so a
   border wall doesn't merge everything that touches it.
2. **Pruning to the largest piece deleted a door.** Keeping only the largest
   skeleton piece per region cut the line through a door between two rooms.
   Pieces are now joined by the shortest safe path between them, and only
   fragments under 8 px are dropped. `n_bridges`, `bridge_px` and `n_pruned`
   are recorded for every map.
3. **RRT\* collision check too coarse.** RRT* checked segments every 0.5 px
   while the independent metric checked every 0.25 px, and the metric caught
   corner pixels the planner had skipped. Both now use 0.25 px.

**Known limitation:** joining the skeleton at the *nearest* pixel can make a
short detour into a corner spur (visible at the start of the two-rooms path).
Shortcutting the entry and exit segments is a week-4 candidate, to be
measured, not assumed.

## Design notes

- **One representation.** A Voronoi diagram is `nearest[H, W]` (int64): the
  flat index of each pixel's nearest site, or -1 if there are no sites.
- **Bit-identical backends.** Every backend compares the packed integer key
  `d2 * H*W + site`, so ties break the same way regardless of visit order or
  thread scheduling. (The int32 Metal kernel compares `(d2, site)`
  lexicographically instead, which gives the same order without 64-bit
  math.) Backends are checked against each other with
  `array_equal`, not a tolerance, which means a speedup in Q1 never comes with
  an accuracy caveat.
- **Error is measured on distance, not labels.** JFA can only choose among the
  sites it propagated to a pixel, so on exact ties it can pick a different, equally
  near site than the oracle. That is still a correct Voronoi assignment. (The
  first version of the property tests asserted label equality and caught this.)
- **One-sided error.** JFA can assign a site that is too far, never one that
  is too near. The tests check that bound directly against the oracle.
- **Two independent exact references.** The brute-force oracle (same tie-break
  rule) is itself checked against SciPy's Euclidean distance transform.
- **Generalized Voronoi diagram sites.** For an occupancy grid, the sites are obstacle
  pixels 4-adjacent to free space. Interior obstacle pixels never win a
  nearest-site query from free space.
- **OpenMP clash.** On macOS, Numba's default OpenMP threading layer and the
  libomp bundled with PyTorch abort the process when both load
  (`OMP: Error #15`). Numba is pinned to its `workqueue` layer; override it
  with `NUMBA_THREADING_LAYER`.
- **Fail loudly.** Unknown config keys, backends and variants raise errors before a
  sweep starts.

## Layout

```
src/voronav/
  config.py            dataclass configs, strict YAML loading
  maps.py              synthetic maps, MovingAI .map + image-mask loaders, boundary sites
  voronoi/
    common.py          representation, tie-break, step schedules, distance
    oracle.py          brute-force exact diagram
    jfa_numpy.py       reference JFA
    jfa_numba.py       multithreaded CPU JFA (dynamic or static row scheduling)
    jfa_torch.py       fused int32 Metal kernel on MPS; eager int64 on CPU / CUDA
  planning/
    roadmap.py         GVD extraction, frame-aware obstacle labels, piece bridging
    grid_search.py     Numba A* / multi-source Dijkstra, 8-connected, no corner cutting
    planners.py        Voronoi roadmap (join / roadmap-weight variants), grid A*,
                       clearance-weighted A*, RRT* (baseline)
    smoothing.py       line-of-sight shortcutting, applied equally to every planner
    metrics.py         length / clearance / smoothness against the exact EDT
  map_sources.py       Q2 map registry: synthetic generators + MovingAI / image / HouseExpo dirs
configs/               debug.yaml (seconds), bench_quick.yaml (~10 s),
                       bench_q1.yaml (full Q1 sweep, heavy), bench_q2_quick.yaml,
                       bench_q2.yaml (full Q2 study), bench_q2_narrow.yaml
scripts/               bench_jfa.py (harness -> JSONL), plot_q1.py (figures),
                       probe_mps.py (why eager MPS was slow), plan_demo.py (Q2 figure),
                       fetch_maps.py (benchmark maps, see docs/DATA.md),
                       bench_plan.py / plot_q2.py / summarize_q2.py (Q2 study)
tests/                 property, cross-backend, loader, config, end-to-end
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,torch]"
.venv/bin/python -m pytest -q                                          # ~9 s
.venv/bin/python scripts/bench_jfa.py --config configs/bench_quick.yaml # ~10 s, light
.venv/bin/python scripts/plan_demo.py --out docs/q2                     # ~2 s
```

The full Q1 study (`configs/bench_q1.yaml`) is about 5,000 diagrams and over
30 minutes of all-core load. On a fanless laptop, run it once, plugged in; it
pauses 2 s between rows to limit thermal-throttling bias:

```bash
.venv/bin/python scripts/bench_jfa.py --config configs/bench_q1.yaml
.venv/bin/python scripts/plot_q1.py runs/bench_q1/results.jsonl --out docs/q1
```

The Q2 study needs the benchmark maps (`scripts/fetch_maps.py`, about 64 MB;
see `docs/DATA.md`). The main sweep takes about 17 minutes and the
narrow-passage supplement about 8, mostly single-core:

```bash
.venv/bin/python scripts/bench_plan.py --config configs/bench_q2.yaml
.venv/bin/python scripts/plot_q2.py runs/bench_q2/results.jsonl --out docs/q2
.venv/bin/python scripts/summarize_q2.py runs/bench_q2/results.jsonl > docs/q2/q2_summary.md
```

## Testing

214 tests run in about 9 s on a laptop with no downloads. Every property test
runs on every available backend (NumPy, Numba, PyTorch CPU, and MPS or CUDA
when present).

- **Oracle:** matches SciPy's EDT exactly; the tie-break is hand-checked;
  chunking has no effect on the result.
- **JFA properties:** a single site gives an exact result; every pixel is
  assigned to a real site; each site maps to itself; error is one-sided;
  degenerate shapes (1×1, 1×N, N×1) are handled; the error rate is pinned on
  fixed seeds, and the extra pass in `jfa+1` never increases it.
- **Cross-backend:** Numba, Torch CPU and Torch MPS match the NumPy reference
  bit for bit on sparse, dense and obstacle maps for every variant. Numba
  gives the same output at any thread count, schedule and chunk size. Both
  MPS paths (the fused Metal kernel and eager int64) are tested separately,
  including on a site lattice full of exact ties.
- **Grid search:** A* path costs match SciPy's csgraph Dijkstra on the same
  graph (an independent oracle). Also covered: no corner cutting, the
  nearest-target Dijkstra, and unreachable goals.
- **Roadmap:** the corridor skeleton is exactly the centerline; the roadmap
  passes through a door between two rooms; there is one connected piece per free
  region; the robot radius is respected; obstacles touching the border are labeled
  separately; bridging and fragment-dropping are checked by hand.
- **Planners:** a corridor end-to-end test with a known answer. A* runs
  straight along the wall (109 px, clearance 2), while Voronoi follows the
  centerline (clearance ≥ 7) and is longer. Every planner's path is
  collision-free under the independent check; A* is never beaten on grid
  length; RRT* is reproducible from its seed.
- **End to end:** the benchmark harness and the plotting script run on the debug config
  and produce well-formed rows and figures.

## References

- Rong & Tan, "Jump Flooding in GPU with Applications to Voronoi Diagram and
  Distance Transform", I3D 2006 (JFA, and the JFA+1 / 1+JFA variants)
- Aurenhammer, "Voronoi Diagrams — A Survey of a Fundamental Geometric Data
  Structure", ACM Computing Surveys 1991
- Choset & Burdick, "Sensor-Based Exploration: The Hierarchical Generalized
  Voronoi Graph", IJRR 2000 (generalized Voronoi diagram roadmaps)
- Sturtevant, "Benchmarks for Grid-Based Pathfinding", IEEE TCIAIG 2012
  (MovingAI maps)
- Karaman & Frazzoli, "Sampling-based Algorithms for Optimal Motion
  Planning", IJRR 2011 (RRT*)
- Attali & Montanvert, "Computing and Simplifying 2D and 3D Continuous
  Skeletons", CVIU 1997; Shaked & Bruckstein, "Pruning Medial Axes", CVIU 1998
  (object-angle pruning)
- Kirillov et al., "Segment Anything", ICCV 2023; Ravi et al., "SAM 2", 2024
