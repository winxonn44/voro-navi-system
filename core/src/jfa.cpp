#include "voronav/jfa.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"

namespace voronav {
namespace {

constexpr int64_t kBigD2 = std::numeric_limits<int64_t>::max();

// Every entry must be kNoSite or a flat index into the grid. Checked once at
// the API boundary so the kernel can divide and square without overflow.
void check_nearest(GridView<const int64_t> nearest, const char* who) {
  const auto hw = static_cast<int64_t>(nearest.size());
  for (const int64_t s : nearest.flat()) {
    if (s < kNoSite || s >= hw) {
      throw std::invalid_argument(std::string(who) +
                                  ": entries must be -1 or a flat index in [0, H*W)");
    }
  }
}

// One pass with step k: out[y, x] = the best of cur's 3x3 candidates at
// offsets {-k, 0, k}, by the (d2, site) tie-break documented in jfa.hpp.
// Mirrors _pixel in jfa_numba.py. All arithmetic is int64: with sides below
// 2^31, |dy|, |dx| < 2^31, so d2 < 2^63.
void jfa_pass(GridView<const int64_t> cur, GridView<int64_t> out, int64_t k) noexcept {
  const int64_t h = cur.height;
  const int64_t w = cur.width;
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      int64_t best_d2 = kBigD2;
      int64_t best_site = kNoSite;
      for (int64_t a = -1; a <= 1; ++a) {
        const int64_t yy = y + a * k;
        if (yy < 0 || yy >= h) {
          continue;
        }
        for (int64_t b = -1; b <= 1; ++b) {
          const int64_t xx = x + b * k;
          if (xx < 0 || xx >= w) {
            continue;
          }
          const int64_t s = cur.data[static_cast<std::size_t>(yy * w + xx)];
          if (s < 0) {
            continue;
          }
          const int64_t sy = s / w;
          const int64_t dy = y - sy;
          const int64_t dx = x - (s - sy * w);
          const int64_t d2 = dy * dy + dx * dx;
          if (d2 < best_d2 || (d2 == best_d2 && s < best_site)) {
            best_d2 = d2;
            best_site = s;
          }
        }
      }
      out.data[static_cast<std::size_t>(y * w + x)] = best_site;
    }
  }
}

}  // namespace

Grid<int64_t> init_nearest(GridView<const uint8_t> sites) {
  Grid<int64_t> out(sites.height, sites.width, kNoSite);
  const auto in = sites.flat();
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (in[i] != 0) {
      out.data()[i] = static_cast<int64_t>(i);
    }
  }
  return out;
}

std::vector<int> step_schedule(int height, int width, Variant variant) {
  // int64 so rounding a side near INT_MAX up to a power of two cannot overflow.
  int64_t n = 1;
  while (n < std::max(height, width)) {
    n *= 2;
  }
  std::vector<int> steps;
  for (int64_t k = n / 2; k >= 1; k /= 2) {
    steps.push_back(static_cast<int>(k));
  }
  switch (variant) {
    case Variant::Jfa:
      break;
    case Variant::JfaPlus1:
      steps.push_back(1);
      break;
    case Variant::JfaPlus2:
      steps.push_back(2);
      steps.push_back(1);
      break;
    case Variant::OnePlusJfa:
      steps.insert(steps.begin(), 1);
      break;
    default:
      throw std::invalid_argument("step_schedule: unknown variant");
  }
  return steps;
}

Grid<int64_t> run_steps(GridView<const int64_t> init, std::span<const int> steps) {
  check_nearest(init, "run_steps");
  Grid<int64_t> cur = to_grid(init);
  Grid<int64_t> nxt(init.height, init.width);
  for (const int k : steps) {
    jfa_pass(cur.view(), nxt.view(), k);
    std::swap(cur, nxt);
  }
  return cur;
}

Grid<int64_t> distance_sq(GridView<const int64_t> nearest) {
  check_nearest(nearest, "distance_sq");
  const int64_t h = nearest.height;
  const int64_t w = nearest.width;
  Grid<int64_t> out(nearest.height, nearest.width);
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      const auto i = static_cast<std::size_t>(y * w + x);
      const int64_t s = nearest.data[i];
      if (s < 0) {
        out.data()[i] = -1;
        continue;
      }
      const int64_t sy = s / w;
      const int64_t dy = y - sy;
      const int64_t dx = x - (s - sy * w);
      out.data()[i] = dy * dy + dx * dx;
    }
  }
  return out;
}

}  // namespace voronav
