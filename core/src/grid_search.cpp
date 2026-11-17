#include "voronav/grid_search.hpp"

#include <array>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace voronav {

namespace {

constexpr std::array<int, kMoves> kDy{-1, 1, 0, 0, -1, -1, 1, 1};
constexpr std::array<int, kMoves> kDx{0, 0, -1, 1, -1, 1, -1, 1};
// right, down, down-left, down-right: covers each unordered neighbor pair once
constexpr std::array<std::size_t, 4> kContactMoves{3, 1, 6, 7};

}  // namespace

void BinaryHeap::push(double key, int64_t value) {
  items_.push_back({key, value});
  std::size_t i = items_.size() - 1;
  while (i > 0) {
    const std::size_t parent = (i - 1) / 2;
    if (items_[parent].key <= items_[i].key) {
      break;
    }
    std::swap(items_[parent], items_[i]);
    i = parent;
  }
}

HeapEntry BinaryHeap::pop() noexcept {
  const HeapEntry top = items_[0];
  const std::size_t last = items_.size() - 1;
  items_[0] = items_[last];
  items_.pop_back();
  std::size_t i = 0;
  for (;;) {
    const std::size_t left = 2 * i + 1;
    if (left >= last) {
      break;
    }
    std::size_t child = left;
    if (left + 1 < last && items_[left + 1].key < items_[left].key) {
      child = left + 1;
    }
    if (items_[i].key <= items_[child].key) {
      break;
    }
    std::swap(items_[i], items_[child]);
    i = child;
  }
  return top;
}

SourceField source_field(GridView<const uint8_t> passable, GridView<const int64_t> labels,
                         std::optional<GridView<const double>> cost_mult) {
  const int h = passable.height;
  const int w = passable.width;
  if (labels.height != h || labels.width != w) {
    throw std::invalid_argument("source_field: labels must have the grid's shape");
  }
  const bool use_w = cost_mult.has_value();
  GridView<const double> mult;
  if (use_w) {
    mult = *cost_mult;
    if (mult.height != h || mult.width != w) {
      throw std::invalid_argument("source_field: cost_mult must have the grid's shape");
    }
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (passable(y, x) != 0 && !(mult(y, x) >= 1.0)) {
          throw std::invalid_argument("cost_mult must be >= 1 on passable pixels");
        }
      }
    }
  }

  const int64_t width = w;
  SourceField out{Grid<double>(h, w, std::numeric_limits<double>::infinity()),
                  std::vector<int64_t>(passable.size(), -1),
                  Grid<int64_t>(h, w),
                  {}};
  Grid<uint8_t> closed(h, w);
  BinaryHeap heap;
  heap.reserve(1024);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (labels(y, x) > 0 && passable(y, x) != 0) {
        out.cost(y, x) = 0.0;
        out.owner(y, x) = labels(y, x);
        heap.push(0.0, static_cast<int64_t>(passable.index(y, x)));
      }
    }
  }
  while (!heap.empty()) {
    const int64_t u = heap.pop().value;
    const auto uy = static_cast<int>(u / width);
    const auto ux = static_cast<int>(u % width);
    if (closed(uy, ux) != 0) {
      continue;  // stale entry (lazy deletion)
    }
    closed(uy, ux) = 1;
    for (std::size_t i = 0; i < kMoves; ++i) {
      const int vy = uy + kDy[i];
      const int vx = ux + kDx[i];
      if (vy < 0 || vy >= h || vx < 0 || vx >= w || passable(vy, vx) == 0) {
        continue;
      }
      double step = 1.0;
      if (i >= kFirstDiagonal) {
        if (passable(vy, ux) == 0 || passable(uy, vx) == 0) {
          continue;
        }
        step = std::numbers::sqrt2;
      }
      if (use_w) {
        step = step * mult(vy, vx);
      }
      if (closed(vy, vx) != 0) {
        continue;
      }
      const double cand = out.cost(uy, ux) + step;
      if (cand < out.cost(vy, vx)) {
        out.cost(vy, vx) = cand;
        const auto v = static_cast<int64_t>(passable.index(vy, vx));
        out.parent[static_cast<std::size_t>(v)] = u;
        out.owner(vy, vx) = out.owner(uy, ux);
        heap.push(cand, v);
      }
    }
  }

  for (int uy = 0; uy < h; ++uy) {
    for (int ux = 0; ux < w; ++ux) {
      const int64_t owner_u = out.owner(uy, ux);
      if (owner_u == 0) {
        continue;
      }
      for (const std::size_t i : kContactMoves) {
        const int vy = uy + kDy[i];
        const int vx = ux + kDx[i];
        if (vy < 0 || vy >= h || vx < 0 || vx >= w) {
          continue;
        }
        const int64_t owner_v = out.owner(vy, vx);
        if (owner_v == 0 || owner_v == owner_u) {
          continue;
        }
        double step = 1.0;
        if (i >= kFirstDiagonal) {
          if (passable(vy, ux) == 0 || passable(uy, vx) == 0) {
            continue;
          }
          step = std::numbers::sqrt2;
        }
        if (use_w) {
          step = step * 0.5 * (mult(uy, ux) + mult(vy, vx));
        }
        out.contacts.u.push_back(static_cast<int64_t>(passable.index(uy, ux)));
        out.contacts.v.push_back(static_cast<int64_t>(passable.index(vy, vx)));
        out.contacts.cost.push_back(out.cost(uy, ux) + step + out.cost(vy, vx));
      }
    }
  }
  return out;
}

}  // namespace voronav
