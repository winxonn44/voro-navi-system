# voronav-core

The optional C++20 core of voronav: the JFA Voronoi engine, GVD roadmap and
grid planners, ported from the Python package and callable from it with
`backend="cpp"`. It is a separate pip package (`voronav-core`) in the same
repository, so `pip install -e .` for the main package stays pure Python and
never needs a compiler.

**Status: checkpoint C1 (serial JFA).** The C++ API has the `Grid` /
`GridView` types, serial JFA (`init_nearest`, `step_schedule`, `run_steps`,
`distance_sq` in `jfa.hpp`) and the brute-force oracle (`brute_force` in
`oracle.hpp`), tested in C++ only. Python bindings for them come in C2; until
then `make_runner(..., backend="cpp")` raises `NotImplementedError`.

## Layout

```
core/
├── CMakeLists.txt          # library, tests, benchmarks, Python module
├── pyproject.toml          # scikit-build-core; distribution "voronav-core"
├── include/voronav/        # public headers: grid.hpp, jfa.hpp, oracle.hpp, version.hpp
├── src/                    # .cpp files matching the headers
├── tests/                  # GoogleTest (ctest)
├── bench/                  # Google Benchmark
└── python/
    ├── bindings.cpp        # pybind11 module voronav_core._core
    └── voronav_core/       # Python package wrapping _core
```

Python imports it as `voronav_core` (the compiled extension is
`voronav_core._core`). It is not `voronav._core` because the main package is
built by hatchling and should not depend on a compiled file inside it.

## Requirements

- A C++20 compiler: Apple Clang 15+ (Xcode), GCC 13+ or Clang 17+.
- CMake ≥ 3.24 and Ninja. On macOS: `brew install cmake ninja` (and
  optionally `ccache`, which the build picks up automatically).
- Network access on the first configure: GoogleTest v1.17.0, Google Benchmark
  v1.9.4 and pybind11 v3.0.1 are fetched with CMake `FetchContent` at pinned
  tags.

No OpenMP: Apple Clang does not ship it. Threads will use `std::jthread`.

## Build and test the C++ alone

From the repository root:

```bash
cmake -S core -B core/build -G Ninja
cmake --build core/build
ctest --test-dir core/build --output-on-failure
```

`compile_commands.json` lands in `core/build/` for clangd and clang-tidy.
Run the benchmarks with `core/build/bench/core_bench`.

CMake options (all `-D...=ON/OFF`):

| Option | Default | Meaning |
| --- | --- | --- |
| `VORONAV_BUILD_TESTS` | ON (OFF under pip) | GoogleTest suite |
| `VORONAV_BUILD_BENCH` | ON (OFF under pip) | Google Benchmark suite |
| `VORONAV_BUILD_PYTHON` | OFF (ON under pip) | pybind11 module |
| `VORONAV_WERROR` | OFF | `-Werror` (CI turns it on) |
| `VORONAV_SANITIZE` | empty | e.g. `address,undefined` or `thread` |

Warnings are `-Wall -Wextra -Wpedantic -Wconversion -Wshadow` on voronav's own
targets only, never on fetched dependencies.

Sanitizer builds, as CI runs them:

```bash
cmake -S core -B core/build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVORONAV_SANITIZE=address,undefined
cmake --build core/build-asan && ctest --test-dir core/build-asan --output-on-failure
```

## JFA: conventions and C1 results

`run_steps` is a direct port of `_pixel` in `src/voronav/voronoi/jfa_numba.py`,
serial and double-buffered (each pass reads one buffer and writes the other).
Candidates are compared by the pair `(d2, site)` lexicographically, which is
the order of Python's packed key `d2 * H*W + site` without the multiplication,
so it cannot overflow (the packed int64 key overflows past about 46,340 px per
side). All coordinate and distance arithmetic is int64. Inputs to `run_steps`
and `distance_sq` are validated once (every entry `-1` or in `[0, H*W)`), and
a bad one throws `std::invalid_argument`.

**Oracle agreement.** `OracleAgreement.ThousandRandomGrids` runs every variant
against `brute_force` on 1,000 seeded random grids: sides uniform in 1–64 (so
mostly non-square, many odd or 1 wide), plus 80 grids at 96–128 per side; site
densities 0.2%, 1%, 5%, 20% and 60%, Bernoulli per cell (so some grids have no
sites at all). 1,939,437 pixels in total. Every variant keeps the invariants of
`tests/test_jfa_properties.py` on every pixel: assigned iff a site exists, only
to a real site, sites keep themselves, never closer than the true nearest
site. **`jfa+2` gets the distance right on every pixel of every grid** (the
assertion), and also picks the oracle's site everywhere on this set.

Residual error, measured as in Q1 (fraction of pixels whose distance differs
from the oracle's); recorded, not asserted beyond `jfa < 1%` and
`jfa+1 <= jfa`:

| Variant | 0.2% | 1% | 5% | 20% | 60% | All 1,000 grids |
| --- | --- | --- | --- | --- | --- | --- |
| `jfa` | 9.5e-5 | 2.5e-4 | 8.2e-4 | 9.8e-4 | 0 | **3.0e-4** (587 px) |
| `jfa+1` | 1.8e-5 | 3.6e-5 | 5.5e-5 | 0 | 0 | 2.4e-5 (47 px) |
| `jfa+2` | 0 | 0 | 0 | 0 | 0 | 0 |
| `1+jfa` | 1.5e-6 | 3.0e-6 | 0 | 0 | 0 | 1.5e-6 (3 px) |

These are small grids (median side about 32), so the numbers are not directly
comparable to Q1's large-grid rates (plain `jfa` at most 1.6e-4 per
configuration there), but the ordering is the same: `1+jfa` beats `jfa+1` at
the same cost, and `jfa+2` is exact.

The test takes 0.7 s in the default RelWithDebInfo build, about 15 s under
ASan+UBSan (Debug) and about 28 s under TSan (Debug) on the M4 Air; the whole
suite is 34 tests.

**First serial timing** (`BM_RunStepsSerial`, one `run_steps` call including
its copy of the input, 1% sites, `jfa+2`; median of 20 repetitions, Apple Clang
17, RelWithDebInfo `-O2`, M4 MacBook Air on AC power):

| Size | Passes | Median | Stddev |
| --- | --- | --- | --- |
| 256² | 10 | 5.5 ms | 0.03 ms |
| 512² | 11 | 28.4 ms | 0.7 ms |

This is the unoptimized baseline of design doc A7 #1: one thread, bounds
checks on every candidate and an int64 division per candidate. Comparisons
with the NumPy, Numba and Metal backends belong to C4.

## Install into Python

Into the same virtual environment as the main package, from the repository
root:

```bash
.venv/bin/pip install -e ./core
.venv/bin/python -c "import voronav_core; print(voronav_core.version())"
.venv/bin/python -m pytest -q tests/test_core.py
```

This is a scikit-build-core editable install: the Python files are used from
`core/python/voronav_core/`, and the compiled `_core` extension is built once
into `core/build-py/` and installed into the venv. **After changing any C++,
run the same `pip install -e ./core` again**; the build directory is kept, so
the rebuild is incremental. `tests/test_core.py` skips the extension tests
when `voronav-core` is not installed.

To uninstall: `.venv/bin/pip uninstall voronav-core`.

## Formatting and static analysis

`.clang-format` (Google style, 100 columns) and `.clang-tidy` (`bugprone-*`,
`performance-*`, `modernize-*`, `cppcoreguidelines-*`, with the exclusions
explained in the file) live in this folder. Neither tool ships with Xcode;
`brew install clang-format` gives the formatter, and clang-tidy comes with
`brew install llvm`.

```bash
clang-format -i core/include/voronav/*.hpp core/src/*.cpp core/tests/*.cpp core/bench/*.cpp core/python/*.cpp
clang-tidy -p core/build core/src/*.cpp core/tests/*.cpp
```

## CI

`.github/workflows/core.yml` (design doc A9):

- **build-test**: macOS 14 arm64 (Apple Clang), Ubuntu 24.04 with GCC and with
  Clang; configure with `-Werror`, build, `ctest`, then install both packages
  and run the full pytest suite.
- **sanitizers**: Ubuntu Clang, ASan+UBSan build and TSan build, each running
  `ctest`.
- **lint**: clang-format check on every C++ file, clang-tidy on changed files.
- **bench-smoke**: the benchmarks at 256² on macOS, to catch crashes.

All jobs use ccache.
