#pragma once

// Jump Flooding (JFA), serial. A port of src/voronav/voronoi/common.py and the
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

#include <cstdint>
#include <span>
#include <vector>

#include "voronav/grid.hpp"

namespace voronav {

inline constexpr int64_t kNoSite = -1;

// The JFA variants of config.VARIANTS: "jfa", "jfa+1", "jfa+2", "1+jfa".
enum class Variant { Jfa, JfaPlus1, JfaPlus2, OnePlusJfa };

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
// Throws std::invalid_argument if init holds anything but kNoSite or an index
// in [0, H * W).
Grid<int64_t> run_steps(GridView<const int64_t> init, std::span<const int> steps);

// Squared distance from each pixel to its assigned site, or -1 where the pixel
// is kNoSite. Same validation as run_steps.
Grid<int64_t> distance_sq(GridView<const int64_t> nearest);

}  // namespace voronav
