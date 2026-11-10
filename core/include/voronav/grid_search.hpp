#pragma once

// The part of planning/grid_search.py the roadmap needs: the binary heap and
// the multi-source nearest-source field (checkpoint C5). A* and the roadmap
// search are C6.
//
// Exact parity: moves, step costs and the heap are ports of the Numba code
// line by line. std::priority_queue would pop equal keys in another order,
// which changes parents and owners. Costs are compared as doubles, so the file
// is compiled with -ffp-contract=off (core/CMakeLists.txt): a fused
// multiply-add rounds differently from Numba's separate operations.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "voronav/grid.hpp"

namespace voronav {

// Grid neighbors in Numba's order: up, down, left, right, then the diagonals
// (up-left, up-right, down-left, down-right).
inline constexpr std::size_t kMoves = 8;
inline constexpr std::size_t kFirstDiagonal = 4;

struct HeapEntry {
  double key = 0.0;
  int64_t value = 0;
};

// The binary min-heap of grid_search._push / _pop. Equal keys come out in the
// order that exact sequence of sift operations produces, which the tests pin.
class BinaryHeap {
 public:
  void reserve(std::size_t n) { items_.reserve(n); }
  void push(double key, int64_t value);
  // The minimum. Undefined on an empty heap.
  HeapEntry pop() noexcept;
  [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return items_.size(); }

 private:
  std::vector<HeapEntry> items_;
};

// Every move (u -> v) between pixels owned by different sources, in scan order
// of u then right, down, down-left, down-right.
struct Contacts {
  std::vector<int64_t> u;
  std::vector<int64_t> v;
  std::vector<double> cost;  // source(u) -> u -> v -> source(v)
};

struct SourceField {
  Grid<double> cost;            // +inf where unreached
  std::vector<int64_t> parent;  // flat [H * W]; -1 at sources and unreached pixels
  Grid<int64_t> owner;          // label of the nearest source; 0 if unreached
  Contacts contacts;
};

// grid_search.source_field: one Dijkstra from every pixel with labels > 0 on a
// passable pixel, with search's move rules (corner_ok = passable). With
// cost_mult, entering a pixel costs the step length times its multiplier.
// Throws std::invalid_argument on a shape mismatch, or if cost_mult is not
// >= 1 on every passable pixel (NaN included).
SourceField source_field(GridView<const uint8_t> passable, GridView<const int64_t> labels,
                         std::optional<GridView<const double>> cost_mult = std::nullopt);

}  // namespace voronav
