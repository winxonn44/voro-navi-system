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
