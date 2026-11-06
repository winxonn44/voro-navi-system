# voronav-core: results log

Measured numbers for the optional C++ core (`core/`, see
[core/README.md](../core/README.md) for the build). The C4 optimization log
comes first; the C2 and C3 sections below it are kept as they were measured. Every row here was
checked bit-identical to the NumPy reference before it was timed: the test
suite runs the `cpp` backend against NumPy with `np.array_equal` (see
"Parity" below), so speedups carry no accuracy caveat.

## Optimization log (checkpoint C4, 2026-11-05)

Design doc A7, one hypothesis at a time, each with a before/after benchmark
on the same inputs and a flame graph; the losers are kept. Every kernel
change kept the output **bit-identical**: all ctest cases (now 69, including
the determinism matrix and the pinned `jfa+2` counterexample) pass in the
normal, ASan+UBSan and TSan builds, and pytest passes (554, 1 skipped)
after the last step.

### Summary

Headline input: 2048², `jfa+1`, 0.1% random point sites, one `run_steps`
call (all 12 passes plus the conversion in and out), median of 20.

| | Time | C4 best is |
| --- | --- | --- |
| **C4 best: C++ `kernel="auto"` (packed + SIMD), 10 threads, dynamic, chunk 4** | **19.1 ms** (harness) / 17.9 ms (Google Benchmark) | |
| Numba, 10 threads, dynamic (same session) | 64.2 ms | **3.4× faster** |
| Numba, 1 thread (same session) | 327.6 ms | 17× faster |
| C++ C3: reference kernel, 10 threads, dynamic (same day) | 58.0 ms | 3.2× faster |
| C++ C2: reference kernel, serial (same day) | 310.4 ms | 17× faster |
| C++ C4 kernel, serial (same session) | 109.8 ms | (3.0× Numba 1 thread) |
| NumPy (same session) | 3,931 ms | 206× faster |
| PyTorch MPS, fused Metal kernel (same session) | **14.3 ms** | **1.34× slower** |

- Across the A6 inputs (1024², 2048², 4096²; `jfa+1` and `jfa+2`; 0.1% and
  1% sites) the C++ core at 10 threads is **2.8–3.7× faster than Numba at 10
  threads** (dynamic, its best schedule), measured in one session.
- **It does not beat the GPU on large grids.** The fused Metal kernel on
  the M4's GPU is 1.33–1.36× faster at 2048² and 4096². At 1024² the CPU
  ties or wins (0.99–1.21×: 5.1–5.5 ms against 5.0–6.7 ms), where the GPU's
  fixed per-pass dispatch cost is a larger share of its time.
- **Distance to the memory-bandwidth floor** (M4: about 120 GB/s): the C3
  kernel moved about 0.9 GB per call at 2048² (12 passes over two 32 MB int64
  buffers, plus the input copy, zero-fill and validation), a floor of about
  7.7 ms, and its best run (58 ms) was about **7.5× above it**. The packed
  layout halves the traffic to about 0.5 GB (16 MB int32 buffers, plus the
  conversions), a floor of about 4.3 ms; 17.9–19.1 ms is about **4–4.5×
  above that** (about 27 GB/s achieved). It is still compute-bound: the
  flame graph puts about 90% of busy samples in the vectorized candidate
  loop, and serial-to-10-thread scaling is 5.4–5.9×, which a bandwidth-bound
  kernel would not get on this chip.

| # | Hypothesis (A7) | Result at 2048² `jfa+1` 0.1%, serial / 10 threads | Verdict |
| --- | --- | --- | --- |
| H1 | Interior/border split removes bounds-check branches | 310.4 → 292.5 ms (1.06×) / 58.0 → 58.9 ms (0.99×) | Kept (small; the structure H2–H4 build on) |
| H2 | Division-free int16 (y, x) layout, half the bytes | 292.5 → 234.2 ms (1.25×) / 58.9 → 40.2 ms (1.47×) | Kept |
| H3 | Chunk size and static vs dynamic | chunks 1–8 within 4% of chunk 4; static 15–25% slower at 10 threads | Rejected (keep dynamic, chunk 4) |
| H4 | SIMD: Clang's vectorizer (then NEON if needed) | 234.2 → 107.9 ms (2.17×) / 40.2 → 20.1 ms (2.00×) by asking Clang to vectorize; wider widths: no change; no NEON | Kept (pragma); hand NEON not needed |
| H5 | User-interactive QoS keeps workers on P-cores | 0.992–1.005× at 4–10 threads | Rejected |
| H6 | (found by the H4 flame graph) the serial parts of a call | 107.4 → 105.5 ms (1.02×) / 20.1 → 17.9 ms (1.12×) | Kept |

The order differs from the design doc's in one place: SIMD (H4) ran before
the chunk sweep (H3) was used to pick defaults, because vectorizing changes
the cost of a row by 2×, and the sweep should be done on the kernel that
ships. H3 was then measured on the final kernel. H6 was not in the plan: the
10-thread flame graph after H4 showed it.

### How each step was measured

- **Machine:** Apple M4 MacBook Air (4 P + 6 E cores), 16 GB, macOS 26.2,
  **on AC power** for every row (`pmset -g batt`: "AC Power", 99%; the
  scripts record it per row and refuse to run on battery). Fanless: every
  configuration ran as its own process with a short warm-up and a 3 s
  cool-down; all of C4's measurements together were under an hour of
  intermittent load.
- **Build:** CMake `Release`, Apple Clang 17.0.0, `-O3 -DNDEBUG`, no
  `-mcpu`. Each step's binary was kept (`core/build-c4-bins/`, ignored by
  git); every JSONL row records the binary, compiler, flags, git commit, a
  hash of the uncommitted `core/` diff when it ran, and the power state.
- **Benchmark:** `BM_JfaKernel` (core/bench/bench_jfa.cpp), driven by
  `scripts/bench_core_kernels.py`: wall-clock time of one `run_steps` call
  (passes plus the input copy or conversion and buffer allocation; seeding
  is outside), 20 single-call repetitions after a 0.3 s warm-up; tables show
  the median with (min–max) in ms. The inputs are the same seeded grids for
  every variant (Bernoulli sites, `std::mt19937_64` seed 0).
- **Profiles:** [samply](https://github.com/mstange/samply) 0.13.1 (sampling
  at 1 kHz, no root needed for a launched process) on a `-O3 -g` build of the
  same code, converted to SVG flame graphs by `scripts/flamegraph_samply.py`
  (in [docs/flamegraphs/](flamegraphs/)). The per-row worker functions are
  `noinline` so they show up as frames (one call per row segment: no
  measurable cost).
- **Raw data:** `docs/runs/c4_h1_h2.jsonl`, `c4_h4.jsonl`,
  `c4_h3_chunks.jsonl`, `c4_scaling.jsonl`, `c4_h5_qos.jsonl`,
  `c4_h6.jsonl`; the final cross-backend run is `c4_final_harness.jsonl`.
  `python scripts/summarize_core_c4.py docs/runs/c4_h6.jsonl` prints any of
  them as a table.

### H1: interior/border split (A7 #2): kept, small

**Hypothesis.** Most candidates are in bounds; checking `yy` and `xx` for
each of the 9 costs branches. Splitting each pass into an interior, where all
9 are in bounds, and a border should make the interior branch-free on bounds
and faster.

**Change.** `Kernel::Split` (`pass_rows_split` in src/jfa.cpp): for step k
the in-bounds rows are a per-row property and the in-bounds columns change
only at x = k and x = W − k, so each row is cut into at most three segments
with a fixed candidate set. The interior segment runs a straight-line
9-candidate loop; border segments loop over a precomputed (row, offset) list.
No pixel has a bounds check. Same flat int64 layout and division as the
reference.

**Result** (reference → split):

| Input | Serial: before | after | gain | 10 threads: before | after | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% | 63.2 (62.9–63.5) | 58.4 (58.3–58.6) | **1.08×** | 12.5 (12.3–12.7) | 12.1 (11.8–12.9) | **1.04×** |
| 1024², jfa+1, 1.0% | 107.9 (107.2–108.2) | 106.7 (105.4–108.7) | **1.01×** | 19.2 (18.9–19.8) | 19.0 (18.6–30.6) | **1.01×** |
| 1024², jfa+2, 0.1% | 70.2 (69.9–72.5) | 65.1 (64.9–65.7) | **1.08×** | 14.1 (13.9–15.4) | 13.4 (13.3–14.7) | **1.05×** |
| 1024², jfa+2, 1.0% | 115.6 (115.0–116.3) | 114.4 (113.8–114.8) | **1.01×** | 21.1 (20.6–24.5) | 20.4 (20.1–20.8) | **1.03×** |
| 2048², jfa+1, 0.1% | 310.4 (306.3–320.1) | 292.5 (287.8–294.5) | **1.06×** | 58.0 (57.7–60.9) | 58.9 (57.2–79.3) | **0.99×** |
| 2048², jfa+1, 1.0% | 505.6 (501.9–515.9) | 497.1 (485.5–510.2) | **1.02×** | 87.1 (86.5–102.2) | 85.6 (84.9–86.8) | **1.02×** |
| 2048², jfa+2, 0.1% | 339.4 (337.1–341.8) | 319.6 (315.3–353.5) | **1.06×** | 64.5 (64.0–67.3) | 62.0 (61.4–63.7) | **1.04×** |
| 2048², jfa+2, 1.0% | 536.2 (532.5–537.8) | 528.0 (517.8–531.1) | **1.02×** | 92.5 (92.0–96.3) | 91.3 (90.9–92.0) | **1.01×** |

The noisy 2048² 0.1% 10-thread row (the first run had a 114 ms outlier and a
median of 80 ms) was re-run alone; the table shows the re-run.

**Verdict: kept, but it is not where the time goes.** 6–8% serially at 0.1%
density, 1–2% at 1%, and noise at 10 threads. The flame graphs show why: the
time sits in the per-candidate arithmetic either way
([reference](flamegraphs/c2_reference_serial.svg): 99% in `jfa_pass_rows`;
[split](flamegraphs/h1_split_serial.svg): 99% in `pass_rows_split`), and
bounds checks were cheap, well-predicted branches. The tell is density:
1% sites is 1.6–1.7× slower than 0.1% in both kernels, although every pixel does
the same 9 candidates. More distinct sites make the `s < 0` and "is this
closer" branches unpredictable. So branch **mispredictions**, not bounds
checks, are the branch cost, and they need branch-free code, which H2/H4
provide. The split is kept because the segment structure is what H4's
vectorized loop is built on, and because it is the int64 fallback above
32,767 px.

### H2: division-free int16 (y, x) layout (A7 #3): kept

**Hypothesis.** Each candidate pays an int64 division `s / W` to recover
(y, x). Storing (y, x) as two 16-bit fields in an int32 turns that into a
shift and a mask and halves the bytes per pass. The design doc expected up
to 2× if the kernel were memory-bound; C3's floor analysis predicted it is
compute-bound, so the division should be the main win.

**Change.** `Kernel::Packed`: `run_steps` converts the int64 input to int32
cells `(y << 16) | x` (or −1) once, runs the passes on two int32 buffers,
and converts back once, so the public flat int64 contract is unchanged. A
packed site is a non-negative int32 that orders like the flat index (x < W),
so the `(d2, site)` tie-break compares packed values directly. The squared
distance is computed in uint32: the largest real d2, 2 · 32,766², is below
2³¹, so it cannot overflow and never reaches the `UINT32_MAX` sentinel an
absent site gets (both are `static_assert`ed). The fold is written with
selects instead of branches. Valid for sides ≤ 32,767 (`kPackedMaxSide`);
`Kernel::Auto`, the default, falls back to `Split` above that, and an
explicit `Packed` throws. Tests cover sides of exactly 32,767 (1 × 32,767,
32,767 × 1, 2 × 32,767, 32,767 × 3, with a site at the far end so every
d2 is near the 1-D maximum) and the fallback at 32,768 and above.

**Result** (split → packed, both scalar: measured with the code as it was
at this step, before H4, so without the vectorization request and with the
candidate-list border loop; `-DVORONAV_NO_FORCE_VECTORIZE` builds the closest
equivalent from today's code):

| Input | Serial: before | after | gain | 10 threads: before | after | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% | 58.4 (58.3–58.6) | 48.2 (47.9–49.3) | **1.21×** | 12.1 (11.8–12.9) | 9.0 (8.8–9.5) | **1.34×** |
| 1024², jfa+1, 1.0% | 106.7 (105.4–108.7) | 80.9 (80.0–83.2) | **1.32×** | 19.0 (18.6–30.6) | 13.0 (12.9–13.5) | **1.46×** |
| 1024², jfa+2, 0.1% | 65.1 (64.9–65.7) | 52.8 (52.7–53.1) | **1.23×** | 13.4 (13.3–14.7) | 9.7 (9.6–10.2) | **1.38×** |
| 1024², jfa+2, 1.0% | 114.4 (113.8–114.8) | 85.9 (85.3–86.5) | **1.33×** | 20.4 (20.1–20.8) | 14.0 (13.8–14.4) | **1.46×** |
| 2048², jfa+1, 0.1% | 292.5 (287.8–294.5) | 234.2 (232.6–265.2) | **1.25×** | 58.9 (57.2–79.3) | 40.2 (39.8–40.9) | **1.47×** |
| 2048², jfa+1, 1.0% | 497.1 (485.5–510.2) | 372.1 (370.0–385.4) | **1.34×** | 85.6 (84.9–86.8) | 57.9 (57.6–62.3) | **1.48×** |
| 2048², jfa+2, 0.1% | 319.6 (315.3–353.5) | 251.7 (250.9–254.2) | **1.27×** | 62.0 (61.4–63.7) | 43.4 (42.9–43.9) | **1.43×** |
| 2048², jfa+2, 1.0% | 528.0 (517.8–531.1) | 394.0 (392.5–396.6) | **1.34×** | 91.3 (90.9–92.0) | 61.8 (61.4–62.4) | **1.48×** |

**Verdict: kept.** 1.2–1.3× serially, 1.3–1.5× at 10 threads (1.25× and
1.47× on the headline row). The larger gain at 1% density (where H1 gained
nothing) is the branch-free fold: misprediction cost fell along with the
division. The larger gain at 10 threads is the halved traffic starting to
matter once 10 cores share the memory system. The
[flame graph](flamegraphs/h2_packed_scalar_serial.svg) (today's code built
with `-DVORONAV_NO_FORCE_VECTORIZE`: the same layout and fold, scalar) has
89% of samples in the interior segment and about 1% in the two conversions.

### H4: SIMD (A7 #5): kept, by asking Clang to vectorize

**Hypothesis.** The packed loop is branch-free int32 arithmetic over
contiguous rows, which NEON handles 4 lanes at a time. First check whether
Clang already vectorizes it; hand-write NEON only if a measurement says so.

**What the vectorizer said.** `clang++ -O3 -Rpass=loop-vectorize
-Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`:

- reference kernel (`jfa_pass_rows`): *loop not vectorized: could not
  determine number of loop iterations* / *instruction cannot be vectorized*.
  NEON has no integer division, so the int64 kernels cannot vectorize.
- packed interior loop: **"the cost-model indicates that vectorization is
  not beneficial"**. It could vectorize, and chose not to.
- border loops (candidate lists): not vectorized (runtime trip counts).

**H4a: override the cost model.** `#pragma clang loop vectorize(enable)` on
the interior loop (in `VORONAV_VECTORIZE_LOOP`; a no-op on GCC). Clang then
vectorizes it at width 4 (*vectorized loop (vectorization width: 4,
interleaved count: 1)*). Result (scalar packed → interior vectorized):

| Input | Serial: before | after | gain | 10 threads: before | after | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% | 48.2 (47.9–49.3) | 29.5 (29.2–30.0) | **1.64×** | 9.0 (8.8–9.5) | 5.4 (5.2–5.9) | **1.68×** |
| 1024², jfa+1, 1.0% | 80.9 (80.0–83.2) | 33.5 (33.2–33.8) | **2.41×** | 13.0 (12.9–13.5) | 6.0 (5.8–6.2) | **2.18×** |
| 1024², jfa+2, 0.1% | 52.8 (52.7–53.1) | 31.9 (31.5–32.3) | **1.66×** | 9.7 (9.6–10.2) | 5.7 (5.5–5.8) | **1.70×** |
| 1024², jfa+2, 1.0% | 85.9 (85.3–86.5) | 35.5 (35.0–36.2) | **2.42×** | 14.0 (13.8–14.4) | 6.5 (6.3–6.8) | **2.16×** |
| 2048², jfa+1, 0.1% | 234.2 (232.6–265.2) | 123.2 (122.6–123.6) | **1.90×** | 40.2 (39.8–40.9) | 22.2 (21.9–23.9) | **1.81×** |
| 2048², jfa+1, 1.0% | 372.1 (370.0–385.4) | 143.6 (143.0–144.1) | **2.59×** | 57.9 (57.6–62.3) | 24.8 (24.6–25.7) | **2.34×** |
| 2048², jfa+2, 0.1% | 251.7 (250.9–254.2) | 132.8 (131.7–134.0) | **1.90×** | 43.4 (42.9–43.9) | 23.6 (23.3–24.3) | **1.84×** |
| 2048², jfa+2, 1.0% | 394.0 (392.5–396.6) | 151.8 (151.0–154.7) | **2.60×** | 61.8 (61.4–62.4) | 26.3 (25.9–26.8) | **2.35×** |

**H4b: vectorize the border too.** After H4a the scalar border segments
were a large share again: the first passes of a JFA run are almost all
border (at k = N/2 there is no interior at all). The border loop needed a
candidate list only because some rows and columns are out of bounds. Now a
missing row points at `none`, a row of W absent sites (−1), which the fold
skips exactly as the reference skips an out-of-bounds row, and the column
offsets a segment may use (left, right, both, neither) are template
parameters (`packed_segment<kLeft, kRight>`). Every segment runs the same
vectorized loop with 3–9 candidates. Result (H4a → H4b):

| Input | Serial: before | after | gain | 10 threads: before | after | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% | 29.5 (29.2–30.0) | 25.7 (25.3–26.1) | **1.15×** | 5.4 (5.2–5.9) | 4.8 (4.6–5.2) | **1.11×** |
| 1024², jfa+1, 1.0% | 33.5 (33.2–33.8) | 25.1 (24.9–25.3) | **1.33×** | 6.0 (5.8–6.2) | 4.8 (4.6–5.2) | **1.24×** |
| 1024², jfa+2, 0.1% | 31.9 (31.5–32.3) | 27.6 (27.2–27.7) | **1.16×** | 5.7 (5.5–5.8) | 5.3 (5.0–5.7) | **1.08×** |
| 1024², jfa+2, 1.0% | 35.5 (35.0–36.2) | 27.5 (27.2–27.7) | **1.29×** | 6.5 (6.3–6.8) | 5.2 (5.0–5.5) | **1.24×** |
| 2048², jfa+1, 0.1% | 123.2 (122.6–123.6) | 107.9 (107.3–108.4) | **1.14×** | 22.2 (21.9–23.9) | 20.1 (19.8–20.6) | **1.11×** |
| 2048², jfa+1, 1.0% | 143.6 (143.0–144.1) | 107.9 (107.4–108.6) | **1.33×** | 24.8 (24.6–25.7) | 20.1 (19.9–20.7) | **1.23×** |
| 2048², jfa+2, 0.1% | 132.8 (131.7–134.0) | 116.9 (116.6–117.4) | **1.14×** | 23.6 (23.3–24.3) | 21.5 (21.2–22.0) | **1.10×** |
| 2048², jfa+2, 1.0% | 151.8 (151.0–154.7) | 117.1 (116.6–117.7) | **1.30×** | 26.3 (25.9–26.8) | 21.5 (21.3–22.6) | **1.22×** |

**H4c: wider vectors or interleaving: rejected.** `vectorize_width(8)`,
`vectorize_width(16)`, `interleave_count(2)` and both: 105–109 ms serial and
20.4–22.0 ms at 10 threads against 105 / 21.5 ms for the default width 4.
That is the same within noise (10 repetitions each at 2048²), so the code keeps
the plain `vectorize(enable)`.

**Hand-written NEON: not done.** The compiler's 4-lane code already runs
the whole kernel; after H4b the [serial flame
graph](flamegraphs/h4_packed_simd_serial.svg) has 97% of samples in the
three `packed_segment` instantiations, which are straight-line vector code.
Intrinsics could at most shave instruction choices in that loop, and width
experiments (H4c) found nothing to gain. It is left as an analysed non-win
that stays open.

**Verdict: kept.** H4a + H4b together: **2.0–3.5× over scalar packed**
(2.17× serial, 2.0× at 10 threads on the headline input). Times no longer
depend on density at all (107.9 ms at 0.1% and at 1%), because nothing
branches on the data any more.

### H3: schedule and chunk size (A7 #4): rejected (defaults confirmed)

**Hypothesis.** The Numba study found dynamic scheduling beats static on the
4P + 6E chip. Does that carry over to C++, and is the default chunk of 4
rows the right one now that a row is 2–3× cheaper?

**Result** (final kernel through H4b, `jfa+1` 0.1%, median ms; static
ignores the chunk, so its 6 columns are 6 independent runs of the same
configuration and show the noise level):

| Size, threads | dynamic, chunk 1 | 2 | 4 | 8 | 16 | 32 | static (6 runs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024², 4 | 7.20 | 7.15 | 7.20 | 7.06 | 7.07 | 7.11 | 7.02–7.37 |
| 1024², 10 | 4.91 | 4.91 | **4.88** | 4.96 | 5.17 | 5.27 | 6.00–7.71 |
| 2048², 4 | 31.80 | 30.85 | 30.67 | 30.73 | 30.72 | 30.51 | 30.64–32.15 |
| 2048², 10 | 20.85 | **20.18** | 20.29 | 20.35 | 20.62 | 21.35 | 23.77–25.49 |

Thread scaling at 2048² (median ms; speedup over serial in parentheses):

| Threads | Reference, static | Reference, dynamic | C4 kernel, static | C4 kernel, dynamic |
| --- | --- | --- | --- | --- |
| 1 | 311.2 | 311.2 | 108.8 | 108.8 |
| 2 | 160.5 (1.94×) | 160.3 (1.94×) | 56.7 (1.92×) | 56.9 (1.91×) |
| 4 | 86.6 (3.59×) | 83.7 (3.72×) | 30.8 (3.53×) | 30.6 (3.56×) |
| 6 | **89.4 (3.48×)** | 73.0 (4.26×) | **30.5 (3.57×)** | 25.5 (4.27×) |
| 8 | 72.5 (4.29×) | 64.1 (4.86×) | 24.0 (4.53×) | 22.1 (4.92×) |
| 10 | 70.9 (4.39×) | 59.2 (5.26×) | 27.8 (3.91×) | **20.2 (5.39×)** |

**Verdict: no change; the Numba conclusion carries over.** Chunk size does
not matter at 4 threads (all within 4%, all P-cores), and at 10 threads
chunks of 1–8 rows tie while 16–32 are 2–8% slower (fewer, larger claims
balance worse across P and E cores at the end of a pass). Static is 15–25%
slower than dynamic at 10 threads and still stalls at 6 threads (4 P-core
blocks wait for 2 E-core blocks), with the faster kernel as with the old
one. The defaults stay: dynamic, chunk 4.

### H5: user-interactive QoS for the workers (A7 #6): rejected

**Hypothesis.** macOS cannot pin threads, but QoS steers placement:
`std::thread` workers start at `QOS_CLASS_DEFAULT` (measured: class 21),
while a terminal-launched main thread is already `USER_INTERACTIVE` (33).
Raising the workers to user-interactive might keep them on P-cores and fix
the 6-thread static stall.

**Change.** `ThreadPool(n, WorkerQos::UserInteractive)` calls
`pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` in each worker
(a no-op off macOS); a test checks the workers report class 33 and the
caller is untouched. Benchmark `BM_JfaQos`, final kernel through H4b, `jfa+1`
0.1%:

| Size, schedule | 4 threads | 6 | 8 | 10 |
| --- | --- | --- | --- | --- |
| 1024², static: default → UI | 7.25 → 7.30 | 7.17 → 7.16 | 5.78 → 5.81 | 5.98 → 5.95 |
| 1024², dynamic | 7.08 → 7.08 | 5.97 → 6.02 | 5.21 → 5.22 | 4.82 → 4.85 |
| 2048², static | 30.69 → 30.74 | 30.35 → 30.54 | 24.20 → 24.25 | 23.45 → 23.61 |
| 2048², dynamic | 30.63 → 30.68 | 25.40 → 25.45 | 22.04 → 22.04 | 20.16 → 20.17 |

**Verdict: rejected.** Every ratio is 0.992–1.005×, and the static 6-thread
stall is unchanged. A busy DEFAULT-class thread is already placed on a free
P-core; with more runnable threads than P-cores, some must run on E-cores
whatever their class. The option stays in the C++ API (tested, two lines)
but is not exposed in Python.

### H6: the serial part of a call (found in the H4 flame graph): kept

**Hypothesis.** The [10-thread flame graph after
H4](flamegraphs/h4_packed_simd_10t.svg) shows the calling thread spending
about 17% of `run_steps` outside the parallel passes: `check_nearest` (the
input validation, a serial read of the whole 32 MB input, about 8%) and the
zero-fill of the three new buffers (about 9%). Serially that is 2% of a call,
but at 10 threads Amdahl's law makes it about 3 ms of 20.

**Change.** (1) Validation runs on the pool (`rows_valid` per row block; a
relaxed atomic flag, ordered by the pool's completion barrier) when a pool is
given. (2) `Grid(h, w, kUninitialized)` allocates without zeroing (a
`std::vector` with a default-initializing allocator), used for the packed
buffers and the output, every cell of which a parallel pass writes before
anything reads it. The pages are now first touched by the parallel passes.
The reference and split kernels keep their C1–C3 allocation. New tests: a
bad entry in the first, middle or last cell is still rejected for every
kernel, schedule and chunk; uninitialized grids have the right shape.

**Result** (before → after, final kernel):

| Input | Serial: before | after | gain | 4 threads: before | after | gain | 10 threads: before | after | gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% | 25.6 (25.3–26.0) | 25.0 (24.7–25.3) | **1.02×** | 7.1 (7.0–7.3) | 6.7 (6.6–6.9) | **1.06×** | 4.8 (4.6–5.0) | 4.3 (4.1–4.9) | **1.13×** |
| 1024², jfa+1, 1.0% | 25.2 (25.0–25.6) | 24.6 (24.4–25.0) | **1.02×** | 7.1 (7.0–7.5) | 6.7 (6.6–7.4) | **1.06×** | 4.8 (4.5–5.0) | 4.3 (4.1–4.6) | **1.12×** |
| 1024², jfa+2, 0.1% | 27.6 (27.3–27.8) | 26.8 (26.6–27.2) | **1.03×** | 7.7 (7.6–8.0) | 7.3 (7.2–7.4) | **1.06×** | 5.2 (4.9–5.4) | 4.7 (4.5–5.0) | **1.11×** |
| 1024², jfa+2, 1.0% | 27.4 (27.2–27.6) | 26.9 (26.6–27.4) | **1.02×** | 7.7 (7.6–7.8) | 7.3 (7.2–7.5) | **1.05×** | 5.1 (4.9–5.3) | 4.6 (4.5–5.0) | **1.11×** |
| 2048², jfa+1, 0.1% | 107.4 (106.2–108.0) | 105.5 (104.7–106.4) | **1.02×** | 30.8 (30.5–31.0) | 28.9 (28.6–29.1) | **1.07×** | 20.1 (19.9–20.4) | 17.9 (17.8–20.0) | **1.12×** |
| 2048², jfa+1, 1.0% | 108.3 (107.5–108.8) | 105.7 (105.1–107.5) | **1.02×** | 30.7 (30.5–31.1) | 29.4 (29.0–32.3) | **1.04×** | 20.1 (19.9–23.9) | 18.0 (17.8–19.2) | **1.12×** |
| 2048², jfa+2, 0.1% | 117.5 (117.1–118.1) | 114.7 (113.9–115.9) | **1.03×** | 33.1 (32.9–33.2) | 31.3 (31.1–31.9) | **1.06×** | 21.5 (21.0–25.7) | 19.5 (19.3–20.2) | **1.10×** |
| 2048², jfa+2, 1.0% | 117.1 (116.6–120.9) | 114.7 (113.7–115.9) | **1.02×** | 33.0 (32.9–33.8) | 31.5 (31.3–32.7) | **1.05×** | 21.1 (20.8–22.4) | 19.3 (19.1–20.5) | **1.09×** |

**Verdict: kept.** 2–3% serial, 4–7% at 4 threads, 9–13% at 10 threads, which
is the pattern Amdahl predicts. [After](flamegraphs/h6_final_10t.svg), the
pool's validation and the two conversions are 1–3% each, spread over all
threads, and the [serial profile](flamegraphs/h6_final_serial.svg) is 97%
`packed_segment`.

### What is left

On the [final 10-thread profile](flamegraphs/h6_final_10t.svg) the busy
samples are the vectorized candidate loop. Most of the rest is the calling
thread in `__ulock_wait` at the end of each pass, waiting for the last rows
from E-cores. That cost comes from 12 barriers on a 4P + 6E chip. Cutting it
would take fewer barriers (fusing passes with halo exchange) rather than
faster pixels. The GPU stays ahead at 2048² and above. The next step there is
design doc A7 #7 (metal-cpp), not more CPU tuning.

### Final comparison (all backends, one session)

`python scripts/bench_jfa.py --config configs/bench_core_c4.yaml`, then
`bench_core_c4_numpy.yaml` and `bench_core_c4_4096.yaml`, back to back on AC
power (2026-11-05, 19:51–20:00): 20 timed runs after 2 warm-ups, 2 s
cool-downs, median ms (min–max). C++ is `voronav_core` 0.0.1 from `pip
install -e ./core` (CMake Release, Apple Clang 17, `-O3 -DNDEBUG`),
`kernel="auto"`. Numba 0.68.0 (workqueue), PyTorch 2.14.1 (MPS, fused int32
Metal kernel), NumPy 2.5.4, Python 3.14.6. Timed region: `make_runner`'s
`run()`. Raw rows: `docs/runs/c4_final_harness.jsonl`.

| Input (passes) | NumPy | Numba 1T | Numba 10T static | Numba 10T dynamic | MPS (Metal) | C++ 1T | C++ 10T static | **C++ 10T dynamic** | vs Numba 10T dyn | vs MPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024², jfa+1, 0.1% (11) | 783.0 (745.7–853.1) | 73.0 (72.1–74.3) | 19.1 (18.7–20.1) | 15.4 (15.0–16.8) | 5.0 (4.7–6.3) | 24.9 (24.1–25.5) | 5.8 (5.6–7.6) | **5.1 (4.5–6.2)** | **3.04×** | 0.99× |
| 1024², jfa+1, 1.0% (11) | – | 72.6 (70.9–74.1) | 19.1 (18.7–20.7) | 15.3 (15.2–17.1) | 6.3 (5.2–6.5) | 24.6 (23.9–25.6) | 5.9 (5.6–7.5) | **5.4 (4.7–7.6)** | **2.84×** | 1.16× |
| 1024², jfa+2, 0.1% (12) | – | 80.2 (78.4–81.4) | 20.9 (20.5–21.8) | 16.7 (16.6–18.1) | 6.3 (5.7–7.1) | 27.0 (25.9–27.6) | 6.3 (6.0–7.3) | **5.2 (5.0–6.7)** | **3.20×** | 1.21× |
| 1024², jfa+2, 1.0% (12) | – | 79.8 (78.6–81.2) | 21.1 (20.6–21.7) | 17.2 (16.7–19.3) | 6.7 (6.2–7.3) | 27.2 (26.2–28.0) | 6.5 (6.1–8.3) | **5.5 (5.1–7.1)** | **3.10×** | 1.21× |
| 2048², jfa+1, 0.1% (12) | 3931.3 (3907.9–4036.8) | 327.6 (325.9–333.0) | 80.8 (79.9–81.7) | 64.2 (64.0–67.6) | 14.3 (14.2–19.7) | 109.8 (107.9–111.1) | 22.6 (22.2–24.3) | **19.1 (18.9–19.5)** | **3.36×** | 0.75× |
| 2048², jfa+1, 1.0% (12) | – | 329.2 (328.6–330.4) | 80.9 (80.2–81.5) | 64.5 (64.0–69.1) | 14.3 (14.1–14.4) | 110.2 (109.0–110.9) | 22.6 (22.2–24.0) | **19.2 (19.0–21.4)** | **3.36×** | 0.75× |
| 2048², jfa+2, 0.1% (13) | – | 358.9 (357.4–369.3) | 87.9 (86.3–88.9) | 76.5 (69.8–83.8) | 15.4 (15.3–16.0) | 119.3 (117.8–120.0) | 24.3 (23.9–24.9) | **20.6 (20.4–22.1)** | **3.71×** | 0.75× |
| 2048², jfa+2, 1.0% (13) | – | 405.8 (358.4–456.1) | 88.0 (87.1–89.3) | 70.2 (69.8–74.8) | 15.5 (15.3–16.1) | 119.8 (118.7–132.2) | 24.4 (24.0–25.4) | **20.6 (20.4–22.1)** | **3.41×** | 0.75× |
| 4096², jfa+1, 0.1% (13) | – | – | – | 281.9 (279.7–299.0) | 60.3 (60.0–60.6) | 480.3 (479.4–481.0) | – | **81.7 (81.1–93.3)** | **3.45×** | 0.74× |
| 4096², jfa+1, 1.0% (13) | – | – | – | 296.8 (282.7–314.1) | 60.4 (60.3–60.7) | 482.5 (480.5–494.3) | – | **81.9 (81.1–90.4)** | **3.63×** | 0.74× |
| 4096², jfa+2, 0.1% (14) | – | – | – | 313.7 (303.1–327.4) | 65.1 (64.9–65.2) | 518.3 (517.1–520.3) | – | **87.2 (86.8–104.9)** | **3.60×** | 0.75× |
| 4096², jfa+2, 1.0% (14) | – | – | – | 320.7 (304.8–330.0) | 65.1 (64.7–65.3) | 518.5 (516.7–533.1) | – | **87.5 (86.7–89.9)** | **3.67×** | 0.74× |

"vs Numba 10T dyn" is Numba's 10-thread dynamic median over C++'s (above 1:
C++ faster); "vs MPS" is the same against the GPU (below 1: the GPU is
faster). Numba 1T is its dynamic-schedule row; the static row at 1 thread
runs the same loop and is within 2%, except at 2048² `jfa+2` 1%, where the
dynamic row is the noisy one (358–456 ms) and static gave 363.9 ms. NumPy
ran on the headline input only (about 4 s per call at 2048²). At 4096² the
two int64 buffers are 128 MB each, and only the fastest configurations ran
(C++ static and Numba 1 thread / static were skipped to keep the load short).

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
of `std::thread` workers created once and reused for every pass;
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
that creates and joins `n − 1` `std::thread`s around every pass, using the
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

(Kept as written at C3; the results are in "Optimization log" above.)

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
