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
| Q1 | How does parallel JFA scale across backends, threads and grid size, and how much error does each variant leave? | **first results in** (below); GPU and scheduling follow-ups open |
| Q2 | Voronoi roadmap vs grid A* vs RRT*: clearance, length, time, success | planned (weeks 3–4) |
| Q3 | How does segmentation error carry through to path safety? | planned (week 5) |
| Q4 | SAM variants on-device: latency vs mask quality vs path quality | planned (week 6) |
| Q5 | High-dimensional Voronoi (k-means IVF) for place recognition: recall vs dimension | stretch (week 7) |

## Q1 first results (measured, Apple M4, 10 cores)

`configs/bench_q1.yaml` as committed: 4 grid sizes × 3 site densities × 4
variants × 9 backend configs = 432 rows, median of 5 runs after 2 warmups. Raw
data: `docs/runs/q1_results.jsonl`.

![JFA time per diagram](docs/q1/q1_throughput.png)

| `jfa+1`, 2048², 0.001 sites/px | median time | vs NumPy |
| --- | --- | --- |
| NumPy reference | 4165 ms | 1× |
| PyTorch CPU | 2848 ms | 1.5× |
| PyTorch MPS (GPU) | 1478 ms | 2.8× |
| Numba, 1 thread | 334 ms | 12× |
| **Numba, 8 threads** | **90 ms** | **46×** |

**The GPU loses to the CPU, and the cause is still a hypothesis.** The MPS
kernel is 16× slower than multithreaded Numba. Two likely causes, neither
verified yet: the int64 keys may be emulated on Apple GPUs (Metal is natively
32-bit), and each pass allocates 9 full-grid temporaries where Numba fuses
everything into one loop over pixels. The next experiment is an int32 encoding
(separate int16 coordinates plus a float32 distance) with a fused kernel.

![Numba strong scaling](docs/q1/q1_scaling.png)

**Scaling follows the M4's core layout, not Amdahl's law.** Speedup is nearly
linear up to 4 threads (3.4×), then falls back at 6 threads (3.1×) and peaks at
3.7× at 8. The M4 has 4 performance cores and 6 efficiency cores. Once work
spreads to efficiency cores, statically split rows finish unevenly and the
slowest thread sets the pace. The Amdahl fit (p = 0.83) is included only to show
that a single parallel fraction cannot describe this curve. Dynamic scheduling
(smaller chunks of rows) is the next step.

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
  thread scheduling. Backends are checked against each other with
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
    jfa_numba.py       multithreaded CPU JFA
    jfa_torch.py       GPU JFA (MPS / CUDA) + CPU
configs/               debug.yaml (seconds), bench_q1.yaml (full Q1 sweep)
scripts/               bench_jfa.py (harness -> JSONL), plot_q1.py (figures)
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

109 tests run in about 20 s on a laptop with no downloads. Every property test
runs on every available backend (NumPy, Numba, PyTorch CPU, and MPS or CUDA
when present).

- **Oracle:** matches SciPy's EDT exactly; the tie-break is hand-checked;
  chunking has no effect on the result.
- **JFA properties:** a single site gives an exact result; every pixel is
  assigned to a real site; each site maps to itself; error is one-sided;
  degenerate shapes (1×1, 1×N, N×1) are handled; the error rate is pinned on
  fixed seeds, and the extra pass in `jfa+1` never increases it.
- **Cross-backend:** Numba, Torch CPU and Torch MPS match the NumPy reference
  bit for bit on sparse, dense and obstacle maps for every variant, and Numba
  gives the same output at any thread count.
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
