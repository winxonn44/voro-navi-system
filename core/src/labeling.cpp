#include "voronav/labeling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace voronav {

DisjointSet::DisjointSet(std::size_t n) : parent_(n), size_(n, 1) {
  std::iota(parent_.begin(), parent_.end(), int64_t{0});
}

int64_t DisjointSet::find(int64_t x) noexcept {
  auto i = static_cast<std::size_t>(x);
  while (parent_[i] != static_cast<int64_t>(i)) {
    parent_[i] = parent_[static_cast<std::size_t>(parent_[i])];  // path halving
    i = static_cast<std::size_t>(parent_[i]);
  }
  return static_cast<int64_t>(i);
}

bool DisjointSet::unite(int64_t a, int64_t b) noexcept {
  a = find(a);
  b = find(b);
  if (a == b) {
    return false;
  }
  auto big = static_cast<std::size_t>(a);
  auto small = static_cast<std::size_t>(b);
  if (size_[big] < size_[small]) {
    std::swap(big, small);
  }
  parent_[small] = static_cast<int64_t>(big);
  size_[big] += size_[small];
  return true;
}

namespace {

// Links each mask pixel to its right and down neighbors and, where diag_ok
// (y, x, y + 1, x +/- 1) allows, to its two lower diagonals; then numbers the
// components by their first pixel in scan order.
template <typename DiagOk>
Labels label_with(GridView<const uint8_t> mask, DiagOk diag_ok) {
  const int h = mask.height;
  const int w = mask.width;
  DisjointSet sets(mask.size());
  const auto id = [&mask](int y, int x) { return static_cast<int64_t>(mask.index(y, x)); };
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (mask(y, x) == 0) {
        continue;
      }
      if (x + 1 < w && mask(y, x + 1) != 0) {
        sets.unite(id(y, x), id(y, x + 1));
      }
      if (y + 1 >= h) {
        continue;
      }
      if (mask(y + 1, x) != 0) {
        sets.unite(id(y, x), id(y + 1, x));
      }
      if (x + 1 < w && mask(y + 1, x + 1) != 0 && diag_ok(y, x, x + 1)) {
        sets.unite(id(y, x), id(y + 1, x + 1));
      }
      if (x > 0 && mask(y + 1, x - 1) != 0 && diag_ok(y, x, x - 1)) {
        sets.unite(id(y, x), id(y + 1, x - 1));
      }
    }
  }
  Labels out{Grid<int64_t>(h, w), 0};
  std::vector<int64_t> number(mask.size(), 0);  // component root -> label
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (mask(y, x) == 0) {
        continue;
      }
      int64_t& label = number[static_cast<std::size_t>(sets.find(id(y, x)))];
      if (label == 0) {
        label = ++out.count;
      }
      out.labels(y, x) = label;
    }
  }
  return out;
}

}  // namespace

Labels label_components(GridView<const uint8_t> mask, Connectivity connectivity) {
  const bool eight = connectivity == Connectivity::Eight;
  return label_with(mask, [eight](int /*y*/, int /*x*/, int /*to_x*/) { return eight; });
}

Labels reachable_components(GridView<const uint8_t> mask,
                            std::optional<GridView<const uint8_t>> corner_ok) {
  const GridView<const uint8_t> corners = corner_ok.value_or(mask);
  if (corners.height != mask.height || corners.width != mask.width) {
    throw std::invalid_argument("reachable_components: corner_ok must have the mask's shape");
  }
  if (mask.size() == 0) {
    throw std::invalid_argument("reachable_components: empty grid");
  }
  return label_with(mask, [&corners](int y, int x, int to_x) {
    return corners(y, to_x) != 0 && corners(y + 1, x) != 0;
  });
}

std::vector<std::size_t> minimum_spanning_tree(int64_t n, std::span<const WeightedEdge> edges) {
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const WeightedEdge& e = edges[i];
    if (e.u < 0 || e.v < 0 || e.u >= n || e.v >= n) {
      throw std::invalid_argument("minimum_spanning_tree: endpoint out of range");
    }
    if (!(e.weight > 0.0)) {  // also rejects NaN
      throw std::invalid_argument("minimum_spanning_tree: weights must be > 0");
    }
    if (i > 0 && std::tie(edges[i - 1].u, edges[i - 1].v) >= std::tie(e.u, e.v)) {
      throw std::invalid_argument("minimum_spanning_tree: edges must be strictly sorted by (u, v)");
    }
  }
  std::vector<std::size_t> order(edges.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(
      order, [edges](std::size_t a, std::size_t b) { return edges[a].weight < edges[b].weight; });
  DisjointSet sets(static_cast<std::size_t>(n));
  std::vector<std::size_t> accepted;
  for (const std::size_t i : order) {
    if (static_cast<int64_t>(accepted.size()) >= n - 1) {
      break;
    }
    if (sets.unite(edges[i].u, edges[i].v)) {
      accepted.push_back(i);
    }
  }
  std::ranges::sort(accepted);
  return accepted;
}

}  // namespace voronav
