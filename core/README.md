# voronav-core

The optional C++20 core of voronav: the JFA Voronoi engine, GVD roadmap and
grid planners, ported from the Python package and callable from it with
`backend="cpp"`. It is a separate pip package (`voronav-core`) in the same
repository, so `pip install -e .` for the main package stays pure Python and
never needs a compiler.

**Status: checkpoint C0 (scaffolding).** The build, tests, benchmarks, Python
bindings and CI are in place. The C++ API so far is `version()` and the
`Grid` / `GridView` types; there is no JFA yet (that is C1), so
`make_runner(..., backend="cpp")` raises `NotImplementedError`.

## Layout

```
core/
├── CMakeLists.txt          # library, tests, benchmarks, Python module
├── pyproject.toml          # scikit-build-core; distribution "voronav-core"
├── include/voronav/        # public headers: grid.hpp, version.hpp
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
Run the benchmark stub with `core/build/bench/core_bench`.

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
