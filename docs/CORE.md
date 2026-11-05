# voronav-core: results log

Measured numbers for the optional C++ core (`core/`, see
[core/README.md](../core/README.md) for the build). Every row here was
checked bit-identical to the NumPy reference before it was timed: the test
suite runs the `cpp` backend against NumPy with `np.array_equal` (see
"Parity" below), so speedups carry no accuracy caveat.

## Parity (checkpoint C2)

`backend="cpp"` goes through the same `make_runner` as the other backends,
so `tests/test_backends_agree.py` and `tests/test_jfa_properties.py` include
`cpp-cpu` whenever `voronav_core` is installed. `tests/test_cpp_parity.py`
adds a cross-language check: `voronav_core.run_steps` against
`jfa_numpy.run_steps` on 16 fixed shapes (1×1, 1×N, N×1, odd, non-square,
both sides of a power of two) × 6 densities (0 to 1) × 6 seeds, plus 80
random shapes, every variant, comparing the `nearest` arrays themselves (site
indices, not just distances). `step_schedule`, `init_nearest`, `distance_sq`
and `brute_force` are compared the same way.

The parity anchor is the minimal `jfa+2` counterexample from C1 (12×13,
sites 39 and 151): JFA is wrong there, and C++ is wrong **identically** to
Python, the same pixels assigned the same wrong site, for every variant
(`jfa` 10 px, `jfa+1` 6, `jfa+2` 1, `1+jfa` 0).

## First serial timing (checkpoint C2, 2026-11-02)

Serial C++ (`run_steps`, a straight port of the Numba per-pixel kernel, no
optimization yet) against the Python backends.

| Grid | Passes | NumPy | Numba, 1 thread | Numba, 10 threads | PyTorch MPS (Metal) | **C++ serial** | C++ vs Numba 1T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256² | 9 | 19.24 ms | 3.88 ms | 2.08 ms | 1.28 ms | **4.59 ms** | 0.85× |
| 512² | 10 | 136.6 ms | 16.41 ms | 5.25 ms | 5.58 ms | **13.70 ms** | 1.20× |
| 1024² | 11 | 896.7 ms | 75.65 ms | 16.00 ms | 5.66 ms | **70.45 ms** | 1.07× |

Median of 10 timed runs; min / max for the C++ column: 256² 3.30 / 8.42 ms,
512² 13.01 / 18.53 ms, 1024² 69.23 / 71.92 ms. The full rows (all times,
min, throughput, error) are reproduced by the config below.

As expected, serial C++ is about single-thread Numba: both compile the same
loop nest with LLVM. The 256² median is noisy (the spread is 3.3–8.4 ms
over 10 runs, versus 3.75–4.16 ms for Numba) and its minimum is below
Numba's, so read it as a tie, not a loss. Making C++ faster is checkpoint C4
(optimization log); multithreading is C3.

### Methodology

- **Machine:** Apple M4 MacBook Air (10 cores: 4 performance + 6
  efficiency), 16 GB RAM, macOS 26.2. Fanless, so it throttles under
  sustained load. **This run was on battery** (55%), not on power as A6 of
  the design doc asks; treat these as first numbers to re-measure, not final
  ones.
- **Software:** Python 3.14.6, NumPy 2.5.4, Numba 0.68.0, PyTorch 2.14.1,
  voronav-core 0.0.1 built by `pip install -e ./core` (scikit-build-core,
  CMake `Release`: Apple Clang 17.0.0, `-O3 -DNDEBUG`, no `-march`/`-mcpu`
  flags).
- **Input:** random point sites, 0.1% density (`random_sites`, seed 0), the
  same grid for every backend; variant `jfa+1`. Error vs the exact EDT is
  identical across backends (0 at 256² and 512², 1.9e-6 at 1024²).
- **Timed region:** only the JFA passes, i.e. `make_runner`'s `run()`.
  Seeding (`init_nearest`), schedule and device upload happen before timing.
  For C++ the input crosses into C++ zero-copy and the result comes back as
  a NumPy array that owns the C++ buffer, so `run()` includes one input copy
  into the first ping-pong buffer and the allocation of the second (as the
  Python backends allocate theirs), nothing else.
- **Repeats:** 1 warm-up run, then 10 timed runs per configuration (the A6
  target is 20; this is the light "quick" pass), 0.5 s cool-down between
  configurations; report the median, with min/max.
- **Command:**

  ```bash
  python scripts/bench_jfa.py --config configs/bench_core_quick.yaml
  ```

  Rows land in `runs/bench_core_quick/results.jsonl` (one JSON line per
  configuration, with all times and the machine description).

## Thread pool (checkpoint C3, 2026-11-02)

`voronav::ThreadPool` (`core/include/voronav/thread_pool.hpp`) is a fixed set
of `std::jthread` workers created once and reused for every pass;
`run_steps(init, steps, pool, schedule, chunk)` splits each pass over it by
rows. `ThreadPool(n)` spawns `n − 1` workers and the calling thread is the
`n`-th. The two schedules mirror the Numba backend: **static** gives
participant *i* the contiguous block `[H·i/n, H·(i+1)/n)`; **dynamic** has
every participant claim the next `chunk` rows (default 4) from a shared
atomic counter until none are left.

Synchronization is three C++20 atomics on separate 128-byte cache lines (the
M4's line size): a generation counter (release bump to start a pass,
acquire in the workers, so the job is visible), the row counter (relaxed
`fetch_add`: it only partitions rows), and a pending count (each worker's
release `fetch_sub` after its last write, the caller's acquire load of 0).
That last pair is the barrier that makes every write of a pass
happen-before the buffer swap. Idle workers sleep in `std::atomic::wait`
(ulock on macOS, futex on Linux): an idle 10-thread pool used 0.2 ms of CPU
over 3 s. No OpenMP. In Python, `voronav_core.run_steps(..., threads=n)`
uses a pool cached per thread count for the life of the process, with the
GIL released for the whole run.

### Determinism

Double buffering makes each output pixel a function of the previous buffer
alone, so how rows are split cannot change the result. The tests check it
instead of assuming it:

- **GoogleTest** (`core/tests/test_parallel.cpp`): exact `Grid` equality with
  the serial result for threads {1, 2, 7, 10} (plus `hardware_concurrency`
  where it differs; on the M4 it is 10) × {static, dynamic} × chunk {1, 4,
  16} = 24 configurations, on 256 grids (16 shapes including 0×0, 1×1, 1×37,
  41×1, 3×50 and other shapes with fewer rows than threads, × 4 densities from 0
  to 1 × 4 variants): **6,144 parallel runs**, plus 256² and 301×177 grids
  under contention and the pinned 12×13 `jfa+2` counterexample (wrong
  pixel 12 reproduced by every configuration). Grids come from
  `test_random.hpp`, so Linux and macOS test the same inputs.
- **Pool tests** (`core/tests/test_thread_pool.cpp`): every row visited
  exactly once for 5 pool sizes × 2 schedules × 5 chunks × 10 row counts;
  static blocks are exactly the equal contiguous split; dynamic claims are
  aligned chunks; 5,000 back-to-back jobs on one pool; a pass reading rows
  other threads wrote in the previous pass; three threads sharing one pool;
  `rows ≤ 0`; `chunk > rows`; first job published before the workers reach
  their wait; and destroying pools (10 threads, and 100 short-lived ones)
  leaves the process thread count where it started.
- **Sanitizers:** all 54 ctest cases pass in RelWithDebInfo, ASan+UBSan and
  **TSan builds with no reports** (locally, AppleClang 17; CI runs the same
  under Linux clang).
- **pytest:** `tests/test_cpp_parity.py` compares `voronav_core.run_steps` with
  threads {2, 3, 7, 10, all} × {static, dynamic} × chunk {1, 4, 16} against
  `jfa_numpy` on 8 shapes × 4 densities × 4 variants, a 512×384 grid and the
  counterexample; `tests/test_core.py` checks the pool cache (one pool per
  thread count, none for `threads=1`), that another Python thread keeps
  running during a C++ call (the GIL is released), and four Python threads
  sharing one pool.

One bug was caught by these tests before any timing: a one-thread pool did
not reset the dynamic row counter between calls, so its second call
processed no rows. The fix is in `ThreadPool::run`.

### Pool vs spawning threads per pass

The design doc predicted that spawning threads every pass costs tens of
microseconds. `core/bench/bench_pool.cpp` measures it two ways:
`BM_EmptyPass*` times one pass with no work (the fixed cost of starting and
finishing a parallel pass), and `BM_Jfa*` runs a full `jfa+2` `run_steps`
(1% sites, static schedule) with the pool or with a benchmark-only driver
that creates and joins `n − 1` `std::jthread`s around every pass, using the
same kernel (`jfa_pass_rows`). Wall-clock medians of 5 repetitions:

| Threads | Empty pass, pool | Empty pass, spawn | Spawn − pool |
| --- | --- | --- | --- |
| 2 | 0.14 µs | 11.0 µs | 10.9 µs |
| 4 | 0.22 µs | 21.1 µs | 20.8 µs |
| 6 | 3.0 µs | 32.8 µs | 29.8 µs |
| 8 | 13.0 µs | 42.9 µs | 29.9 µs |
| 10 | 20.3 µs | 53.1 µs | 32.8 µs |

| JFA `jfa+2`, static | Passes | Pool | Spawn | Saved per pass |
| --- | --- | --- | --- | --- |
| 256², 4 threads | 10 | 1.54 ms | 1.71 ms | 18 µs (10%) |
| 256², 10 threads | 10 | 1.76 ms | 1.96 ms | 19 µs (10%) |
| 1024², 4 threads | 12 | 34.3 ms | 34.0 ms | within noise (σ 1–2 ms) |
| 1024², 10 threads | 12 | 26.9 ms | 26.8 ms | within noise (σ 0.3–0.4 ms) |

**The prediction holds:** creating and joining threads costs about 5 µs per
thread, 11–53 µs per pass at 2–10 threads, and the pool saves 11–33 µs per
pass. That is 10% of a 256² run, where a pass takes about 0.17 ms, and lost
in the noise at 1024², where a pass takes about 2 ms. At 2–4 threads the
pool's empty pass is almost free because back-to-back passes find the
workers still in the short spin that `atomic::wait` does before sleeping;
at 8–10 threads the cost (13–20 µs) is waking sleeping workers, several of
them on efficiency cores. C4 can count on fixed pass overhead below about
20 µs; it only matters for grids of 256² and smaller.

### Strong scaling

`jfa+1`, 0.1% random point sites (seed 0), median of 10 timed runs after 1
warm-up, 1 s cool-down between configurations; speedup is relative to the
same backend and schedule at 1 thread (C++ 1 thread is the serial path, with
no pool and no schedule). Numba ran in the same session, after C++ at each
size.

**2048²** (12 passes):

| Threads | C++ static | C++ dynamic | Numba static | Numba dynamic |
| --- | --- | --- | --- | --- |
| 1 | 317.7 ms (1.00×) | 317.7 ms (1.00×) | 341.3 ms (1.00×) | 338.6 ms (1.00×) |
| 2 | 166.1 ms (1.91×) | 165.8 ms (1.92×) | 170.9 ms (2.00×) | 170.6 ms (1.98×) |
| 4 | 92.2 ms (3.45×) | 87.7 ms (3.62×) | 91.9 ms (3.71×) | 90.1 ms (3.76×) |
| 6 | **94.6 ms (3.36×)** | 76.2 ms (4.17×) | **101.0 ms (3.38×)** | 79.4 ms (4.26×) |
| 8 | 77.9 ms (4.08×) | 70.2 ms (4.52×) | 82.3 ms (4.15×) | 70.8 ms (4.78×) |
| 10 | 76.0 ms (4.18×) | **63.8 ms (4.98×)** | 82.2 ms (4.15×) | 73.7 ms (4.59×) |

**1024²** (11 passes):

| Threads | C++ static | C++ dynamic | Numba static | Numba dynamic |
| --- | --- | --- | --- | --- |
| 1 | 64.1 ms (1.00×) | 64.1 ms (1.00×) | 79.1 ms (1.00×) | 75.3 ms (1.00×) |
| 2 | 33.9 ms (1.89×) | 33.9 ms (1.89×) | 43.2 ms (1.83×) | 38.7 ms (1.94×) |
| 4 | 19.0 ms (3.37×) | 19.0 ms (3.38×) | 34.4 ms (2.30×) | 24.5 ms (3.07×) |
| 6 | **20.1 ms (3.19×)** | 16.6 ms (3.86×) | 23.6 ms (3.35×) | 18.9 ms (3.97×) |
| 8 | 16.4 ms (3.92×) | 15.4 ms (4.16×) | 23.8 ms (3.33×) | 17.0 ms (4.43×) |
| 10 | 20.0 ms (3.20×)* | 18.4 ms (3.48×)* | 20.4 ms (3.87×) | 15.9 ms (4.75×) |

\* Noisy: that C++ static row had a 38 ms outlier. A re-run of just the C++
1024² rows (20 repeats) gave serial 65.9 ms, static 19.4 / **20.1** / 17.1 /
17.4 ms and dynamic 18.6 / 17.0 / 14.6 / **14.4 ms (4.58×)** at 4 / 6 / 8 / 10
threads: 10 threads matches 8 and does not slow down, and the static dip at 6
appears again.

**Interpretation.**

- **C++ reproduces the 6-thread dip under static scheduling, and dynamic
  fixes it.** Static at 6 threads is slower than at 4 at both sizes
  (2048²: 92.2 → 94.6 ms; 1024²: 19.0 → 20.1 ms, in both runs), exactly as
  the Python study found for Numba (here 91.9 → 101.0 ms at 2048²). With 6
  equal blocks, macOS puts 4 on the performance cores and 2 on efficiency
  cores, and the pass waits for the E-core blocks: each is 1/6 of the rows at
  a fraction of P-core speed, which takes longer than 1/4 of the rows on a
  P-core. Dynamic scheduling lets the P-cores take more rows: 76.2 ms at 6
  threads instead of 94.6, and the curve keeps rising to 4.98× at 10.
- **The curve bends after 4 threads because the extra threads are E-cores.**
  Up to 4 threads speedup is close to linear (3.4–3.6×); threads 5–10 add
  efficiency cores, and 6 of them give roughly another 1.4×. 5× at 10
  threads is about what 4 P + 6 E cores can give for this kernel, not a
  pool limit.
- **C++ dynamic is level with or ahead of Numba dynamic** where the
  measurement is clean: 63.8 vs 73.7 ms at 2048² and 10 threads, 70.2 vs
  70.8 at 8. At 1024², 14.4 ms (re-run) against Numba's 15.9 ms. The serial
  kernels are about equal (C2), so the difference is the runtime: a pool
  that starts a pass in a few microseconds, against Numba's `prange` over
  workers plus an atomic counter.
- **Not memory-bound yet.** A 2048² pass reads one 32 MB int64 buffer and
  writes another; 12 passes are about 0.8 GB of traffic, about 6–7 ms at the
  M4's ~120 GB/s. Even the best 10-thread run (63.8 ms) is about 10× above
  that floor, so the kernel is compute-bound: the int64 division `s / w` for
  each of the 9 candidates and the bounds checks dominate. This sets C4's
  order: the interior/border split (A7 #2) and the division-free layout
  (A7 #3) should matter more than halving memory traffic.

### What C4 starts from

- Benchmarks: `BM_RunStepsSerial/n` and `BM_RunStepsPool/n/threads/schedule/chunk`
  in `core/bench/bench_jfa.cpp`, `BM_EmptyPass*` and `BM_Jfa{Pool,Spawn}` in
  `core/bench/bench_pool.cpp`, and the Python harness with
  `configs/bench_core_threads.yaml` (`bench.cpp_threads`,
  `bench.cpp_schedules`).
- `jfa_pass_rows(cur, out, k, y0, y1)` is public, so an alternative kernel or
  driver can be benchmarked against the current one without touching
  `run_steps`.
- Not done here (A7 #6): the QoS-class experiment
  (`pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` in the
  workers). The schedule/chunk sweep (A7 #4) has only the default chunk of 4
  so far.

### Methodology (C3)

- **Machine:** the same Apple M4 MacBook Air (4P + 6E, 16 GB, macOS 26.2),
  this time **on AC power** (battery 65–67%, charging). Fanless; the runs
  are short (the Python sweep was 90 s of load) with 1 s cool-downs, but
  later rows can still be slightly throttled.
- **Software:** as in C2 (Python 3.14.6, NumPy 2.5.4, Numba 0.68.0 with the
  workqueue threading layer), voronav-core 0.0.1 from `pip install -e
  ./core` (CMake `Release`, Apple Clang 17.0.0, `-O3 -DNDEBUG`). The Google
  Benchmark numbers come from a separate `Release` build
  (`cmake -S core -B core/build-release -DCMAKE_BUILD_TYPE=Release
  -DVORONAV_BUILD_TESTS=OFF`).
- **Timed region:** the Python rows time `make_runner`'s `run()` (all passes,
  plus the input copy and the second buffer's allocation); the Google
  Benchmark rows time one `run_steps` call or one empty pass, wall clock.
- **Commands:**

  ```bash
  python scripts/bench_jfa.py --config configs/bench_core_threads.yaml
  core/build-release/bench/core_bench --benchmark_filter='EmptyPass|BM_Jfa' \
      --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
      --benchmark_min_time=0.2s
  ```
