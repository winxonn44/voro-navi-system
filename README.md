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
| Q2 | Voronoi roadmap vs grid A* vs RRT*: clearance, length, time, success | planned (weeks 3–4) |
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
configs/               debug.yaml (seconds), bench_q1.yaml (full Q1 sweep)
scripts/               bench_jfa.py (harness -> JSONL), plot_q1.py (figures),
                       probe_mps.py (why eager MPS was slow)
tests/                 property, cross-backend, loader, config, end-to-end
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,torch]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_jfa.py --config configs/bench_q1.yaml
.venv/bin/python scripts/plot_q1.py runs/bench_q1/results.jsonl --out docs/q1
```

## Testing

126 tests run in about 20 s on a laptop with no downloads. Every property test
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
- Kirillov et al., "Segment Anything", ICCV 2023; Ravi et al., "SAM 2", 2024
