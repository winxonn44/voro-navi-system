#pragma once

// Jump Flooding (JFA), serial or row-parallel. A port of src/voronav/voronoi/common.py and the
// per-pixel kernel of jfa_numba.py, with bit-identical output.
//
// Representation: nearest[H, W] (int64) holds the flat index y * W + x of the
// nearest site found so far, or kNoSite (-1).
//
// Tie-break invariant: among candidate sites, the one with the smallest pair
// (d2, site) in lexicographic order wins, where d2 is the squared Euclidean
// distance and site the flat index. The Python backends compare the packed key
// d2 * (H * W) + site; because 0 <= site < H * W, that key orders candidates
// exactly as the pair does. Comparing the pair directly gives the same result
// with no multiplication, so it cannot overflow at any grid size (the packed
// int64 key overflows past about 46,340 px per side). The winner is therefore
// independent of the order candidates are visited in, which is what makes
// every backend agree with array_equal rather than a tolerance.
//
// Determinism under threads: a pass reads only `cur` and writes only `out`
// (double buffering), and each output pixel depends on `cur` alone, so any
// split of the rows over any number of threads computes the same `out`.
// The pool's completion barrier orders every write of a pass before the
// buffer swap (thread_pool.hpp). Output is bit-identical for every thread
// count, schedule and chunk size (tests/test_parallel.cpp).

#include <cstdint>
#include <span>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/thread_pool.hpp"

namespace voronav {

inline constexpr int64_t kNoSite = -1;

// The JFA variants of config.VARIANTS: "jfa", "jfa+1", "jfa+2", "1+jfa".
enum class Variant : std::uint8_t { Jfa, JfaPlus1, JfaPlus2, OnePlusJfa };

// Seed grid: each site (any nonzero cell) points at its own flat index, every
// other cell is kNoSite.
Grid<int64_t> init_nearest(GridView<const uint8_t> sites);

// Step lengths N/2, N/4, ..., 1 with N = max(height, width) rounded up to a
// power of two, plus the variant's extra passes:
//   JfaPlus1:   one extra k=1 pass at the end
//   JfaPlus2:   extra k=2 and k=1 passes at the end
//   OnePlusJfa: one k=1 pass before the standard schedule
// A 1x1 (or empty) grid has no standard steps, so Jfa gives {}.
// Throws std::invalid_argument for a variant outside the enum.
std::vector<int> step_schedule(int height, int width, Variant variant);

// Runs one JFA pass per step, double-buffered: each pass reads one buffer and
// writes the other, never both in one. Each pixel takes the best of the 3x3
// candidates at offsets {-k, 0, k} (out-of-bounds and kNoSite ones skipped).
// pool == nullptr runs serially on the calling thread (the C1 path);
// otherwise each pass is split over the pool's threads by rows with
// `schedule` and `chunk` (rows per dynamic claim). The result is the same in
// every case. Throws std::invalid_argument if init holds anything but
// kNoSite or an index in [0, H * W), if chunk < 1, or for Kernel::Packed on a
// grid with a side above kPackedMaxSide.
//
// `kernel` picks the per-pass implementation (checkpoint C4, docs/CORE.md
// "Optimization log"); every kernel gives the same bits:
//   Reference: the C1 kernel, jfa_pass_rows below: flat int64 buffers, a
//              bounds check and an int64 division s / W per candidate.
//   Split:     flat int64, but each row is cut into segments where the set of
//              in-bounds candidates is constant, so the interior (all 9
//              candidates in bounds) runs with no bounds checks at all.
//   Packed:    Split's loop structure over an internal int32 layout that
//              stores a site as (y << 16) | x, so decoding it is a shift and
//              a mask instead of a division, and a buffer is half the size.
//              Converted from / to the flat int64 contract once per call.
//              Only for sides <= kPackedMaxSide; throws above that.
//   Auto:      Packed when both sides fit, else Split (the fallback).
inline constexpr int kDefaultChunk = 4;  // as jfa_numba.DEFAULT_CHUNK
enum class Kernel : std::uint8_t { Auto, Reference, Split, Packed };

// Largest side the Packed layout handles: y and x must fit in 15 bits so the
// packed value stays a non-negative int32 and orders like the flat index, and
// the squared distance (at most 2 * 32766^2 < 2^31) fits in 32 bits.
inline constexpr int kPackedMaxSide = 32767;

// The kernel run_steps uses for an H x W grid: Auto becomes Packed or Split,
// anything else is returned as is.
[[nodiscard]] Kernel resolve_kernel(int height, int width, Kernel kernel) noexcept;

Grid<int64_t> run_steps(GridView<const int64_t> init, std::span<const int> steps,
                        ThreadPool* pool = nullptr, Schedule schedule = Schedule::Dynamic,
                        int chunk = kDefaultChunk, Kernel kernel = Kernel::Auto);

// One pass with step k over rows [y0, y1) of out: the kernel run_steps
// splits across threads. Public for benchmarks that drive passes themselves
// (core/bench); it does not validate cur (run_steps does, once).
void jfa_pass_rows(GridView<const int64_t> cur, GridView<int64_t> out, int64_t k, int y0,
                   int y1) noexcept;

// Squared distance from each pixel to its assigned site, or -1 where the pixel
// is kNoSite. Same validation as run_steps.
Grid<int64_t> distance_sq(GridView<const int64_t> nearest);

}  // namespace voronav
