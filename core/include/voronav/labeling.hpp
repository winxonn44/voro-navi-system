#pragma once

// Connected components and minimum spanning trees for the roadmap (design doc
// A8): ports of what roadmap.py gets from scipy.ndimage.label,
// grid_search.reachable_components and scipy.sparse.csgraph.minimum_spanning_tree.
//
// Numbering: labels are 1, 2, ..., count in order of each component's first
// pixel in row-major scan order, 0 outside the mask. reachable_components
// defines its numbering that way, and scipy.ndimage.label produces the same
// one. The numbers matter: obstacle_ids takes the largest label around a
// frame pixel, and connect_pieces breaks size ties by the lowest label.
//
// Which connectivity: the searches forbid corner cutting, so regions of free
// space are labeled with the search's own move rule (reachable_components:
// 4-connected when corner_ok is the mask itself), never by 8-connectivity,
// which can join two regions no path can cross. Obstacles are 8-connected.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "voronav/grid.hpp"

namespace voronav {

struct Labels {
  Grid<int64_t> labels;  // 0 outside the mask, else 1..count
  int64_t count = 0;
};

enum class Connectivity : std::uint8_t { Four, Eight };

// Union-find with path halving and union by size.
class DisjointSet {
 public:
  explicit DisjointSet(std::size_t n);
  [[nodiscard]] int64_t find(int64_t x) noexcept;
  // Joins the sets of a and b; returns false if they were already one set.
  bool unite(int64_t a, int64_t b) noexcept;

 private:
  std::vector<int64_t> parent_;
  std::vector<int64_t> size_;
};

// scipy.ndimage.label with the plus-shaped (Four) or full 3x3 (Eight) structure.
Labels label_components(GridView<const uint8_t> mask, Connectivity connectivity);

// grid_search.reachable_components: orthogonal neighbors in the mask always
// connect; diagonal neighbors connect only when both corner cells are in
// corner_ok (default: the mask itself, which is 4-connectivity). Throws
// std::invalid_argument on a shape mismatch or an empty grid (the Python
// raises ValueError there too).
Labels reachable_components(GridView<const uint8_t> mask,
                            std::optional<GridView<const uint8_t>> corner_ok = std::nullopt);

struct WeightedEdge {
  int64_t u = 0;
  int64_t v = 0;
  double weight = 0.0;
};

// scipy.sparse.csgraph.minimum_spanning_tree: Kruskal over a stable sort of
// the edges by weight, so among equal weights the edge earlier in CSR order
// (row u, then column v) is tried first. Given that order, which edges are
// accepted depends only on connectivity, so the union-find details need not
// match SciPy's.
//
// `edges` must be strictly increasing in (u, v), with endpoints in [0, n) and
// weights > 0 (+inf is allowed). SciPy stores a zero weight as "no edge", so a
// zero or NaN weight has no meaning here and throws std::invalid_argument.
// Returns the indices into `edges` of the accepted edges, in increasing order.
std::vector<std::size_t> minimum_spanning_tree(int64_t n, std::span<const WeightedEdge> edges);

}  // namespace voronav
