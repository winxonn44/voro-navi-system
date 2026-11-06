#include "voronav/jfa.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/thread_pool.hpp"

namespace voronav {
namespace {

constexpr int64_t kBigD2 = std::numeric_limits<int64_t>::max();

// Every entry must be kNoSite or a flat index into the grid. Checked once at
// the API boundary so the kernel can divide and square without overflow.
// With a pool the rows are checked in parallel (H6: serially this read of
// the whole input was about 8% of a 10-thread run at 2048^2).
[[nodiscard]] bool rows_valid(GridView<const int64_t> nearest, int y0, int y1) noexcept {
  const auto hw = static_cast<int64_t>(nearest.size());
  for (int y = y0; y < y1; ++y) {
    for (const int64_t s : nearest.row(y)) {
      if (s < kNoSite || s >= hw) {
        return false;
      }
    }
  }
  return true;
}

void check_nearest(GridView<const int64_t> nearest, const char* who, ThreadPool* pool = nullptr,
                   Schedule schedule = Schedule::Dynamic, int chunk = kDefaultChunk) {
  bool ok = true;
  if (pool == nullptr) {
    ok = rows_valid(nearest, 0, nearest.height);
  } else {
    // Relaxed is enough: parallel_rows' completion barrier orders every
    // store before our load.
    std::atomic<bool> bad{false};
    auto rows = [nearest, &bad](int y0, int y1) noexcept {
      if (!bad.load(std::memory_order_relaxed) && !rows_valid(nearest, y0, y1)) {
        bad.store(true, std::memory_order_relaxed);
      }
    };
    pool->parallel_rows(nearest.height, schedule, chunk, rows);
    ok = !bad.load(std::memory_order_relaxed);
  }
  if (!ok) {
    throw std::invalid_argument(std::string(who) +
                                ": entries must be -1 or a flat index in [0, H*W)");
  }
}

// Clang's cost model declines to vectorize the packed pixel loop on its own
// (docs/CORE.md, H4); the pragma asks for it, and the loop is written so it
// can be (branch-free selects, no calls). GCC vectorizes or not by its own
// judgement. Building with -DVORONAV_NO_FORCE_VECTORIZE drops the request
// (the scalar packed kernel of H2, for comparison).
#if defined(__clang__) && !defined(VORONAV_NO_FORCE_VECTORIZE)
#define VORONAV_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#else
#define VORONAV_VECTORIZE_LOOP
#endif
// The per-row workers are kept out of line: one call per row segment costs
// nothing measurable, and it keeps them visible as frames in a profile
// (the flame graphs in docs/CORE.md).
#if defined(__GNUC__)
#define VORONAV_NOINLINE [[gnu::noinline]]
#else
#define VORONAV_NOINLINE
#endif

// ---- Split and Packed kernels (C4) ----------------------------------------
//
// For a pass with step k, whether a candidate row y + a*k is in bounds
// depends only on the row, and whether a candidate column x + b*k is depends
// only on which side of the breakpoints k and W - k the pixel's x lies. So
// each row is cut into at most three column segments, inside which the set
// of in-bounds candidates is fixed, and no pixel needs a bounds check.
//   Split (pass_rows_split, flat int64): a segment with all 9 candidates in
//     bounds (the interior) runs a straight-line 9-candidate loop; any other
//     (the border) loops over its precomputed (row, column offset) list.
//   Packed (pass_rows_packed, int32 (y, x)): every segment runs one
//     branch-free, vectorized loop (see there).
// Which candidates are visited, and so the result, is exactly the reference
// kernel's; only the order differs, and the (d2, site) minimum does not
// depend on order (jfa.hpp).
//
// A layout policy supplies the cell type, how to decode a site and how to
// fold one candidate into the running best.

// Flat int64 cells (the public layout), decoded with a division as in the
// reference kernel. Same arithmetic, same comparison.
struct Flat64 {
  using Cell = int64_t;
  struct Best {
    int64_t d2 = kBigD2;
    int64_t site = kNoSite;
  };
  int64_t w;

  void fold(Best& best, int y, int x, Cell s) const noexcept {
    if (s < 0) {
      return;
    }
    const int64_t sy = s / w;
    const int64_t dy = y - sy;
    const int64_t dx = x - (s - sy * w);
    const int64_t d2 = dy * dy + dx * dx;
    if (d2 < best.d2 || (d2 == best.d2 && s < best.site)) {
      best.d2 = d2;
      best.site = s;
    }
  }
  [[nodiscard]] static Cell result(const Best& best) noexcept { return best.site; }
};

// Packed int32 cells: (y << 16) | x, or -1. With sides <= kPackedMaxSide both
// fields fit in 15 bits, so a packed site is a non-negative int32, and since
// x < W, packed order is (y, x) lexicographic order, which is flat-index
// order: the tie-break on the packed value picks the same site. Decoding is
// a shift and a mask. The fold is branch-free (selects), so the interior
// loop can be vectorized: an absent site (-1) gets the sentinel distance
// UINT32_MAX, which never beats a real one (real d2 <= 2 * 32766^2 < 2^31),
// and a running best that is still (UINT32_MAX, -1) is beaten by any real
// site and by no absent one. The arithmetic is unsigned so the garbage
// distance computed for an absent site cannot overflow (it is discarded).
struct Packed32 {
  using Cell = int32_t;
  struct Best {
    uint32_t d2 = UINT32_MAX;
    int32_t site = -1;
  };

  static void fold(Best& best, int y, int x, Cell s) noexcept {
    const int32_t sy = s >> 16;  // -1 for an absent site: harmless, discarded below
    const int32_t sx = s & 0xFFFF;
    const auto dy = static_cast<uint32_t>(y - sy);
    const auto dx = static_cast<uint32_t>(x - sx);
    const uint32_t d2 = s < 0 ? UINT32_MAX : dy * dy + dx * dx;
    const bool better = d2 < best.d2 || (d2 == best.d2 && s < best.site);
    best.d2 = better ? d2 : best.d2;
    best.site = better ? s : best.site;
  }
  [[nodiscard]] static Cell result(const Best& best) noexcept { return best.site; }
};

static_assert(2LL * (kPackedMaxSide - 1) * (kPackedMaxSide - 1) < INT32_MAX,
              "packed d2 must stay below the UINT32_MAX sentinel (and fit in int32)");
static_assert((static_cast<int64_t>(kPackedMaxSide - 1) << 16 | (kPackedMaxSide - 1)) <= INT32_MAX,
              "a packed site must be a non-negative int32");

template <typename L>
VORONAV_NOINLINE void pass_rows_split(const L& layout, const typename L::Cell* cur,
                                      typename L::Cell* out, int h, int w, int k, int y0,
                                      int y1) noexcept {
  using Cell = typename L::Cell;
  using Best = typename L::Best;
  const auto row_ptr = [cur, w](int y) noexcept {
    return cur + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(w);
  };
  // Column segments [bounds[i], bounds[i + 1]): the breakpoints are x = k
  // (x - k in bounds from here) and x = W - k (x + k in bounds before here).
  const int lo = std::clamp(std::min(k, w - k), 0, w);
  const int hi = std::clamp(std::max(k, w - k), 0, w);
  const std::array<int, 4> bounds{0, lo, hi, w};

  for (int y = y0; y < y1; ++y) {
    std::array<const Cell*, 3> rows{};
    int n_rows = 0;
    // int64 so y + k cannot overflow on any grid int can describe.
    for (const int64_t yy : {int64_t{y} - k, int64_t{y}, int64_t{y} + k}) {
      if (yy >= 0 && yy < h) {
        rows[static_cast<std::size_t>(n_rows++)] = row_ptr(static_cast<int>(yy));
      }
    }
    Cell* const dst = out + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(w);

    for (std::size_t seg = 0; seg < 3; ++seg) {
      const int xs = bounds[seg];
      const int xe = bounds[seg + 1];
      if (xs >= xe) {
        continue;
      }
      // Constant over the segment by construction, so test its first pixel.
      std::array<int, 3> offs{};
      int n_cols = 0;
      for (const int off : {-k, 0, k}) {
        if (int64_t{xs} + off >= 0 && int64_t{xs} + off < w) {
          offs[static_cast<std::size_t>(n_cols++)] = off;
        }
      }

      if (n_rows == 3 && n_cols == 3) {
        // Interior: all 9 candidates in bounds.
        const Cell* const r0 = rows[0];
        const Cell* const r1 = rows[1];
        const Cell* const r2 = rows[2];
        for (int x = xs; x < xe; ++x) {
          Best best;
          layout.fold(best, y, x, r0[x - k]);
          layout.fold(best, y, x, r0[x]);
          layout.fold(best, y, x, r0[x + k]);
          layout.fold(best, y, x, r1[x - k]);
          layout.fold(best, y, x, r1[x]);
          layout.fold(best, y, x, r1[x + k]);
          layout.fold(best, y, x, r2[x - k]);
          layout.fold(best, y, x, r2[x]);
          layout.fold(best, y, x, r2[x + k]);
          dst[x] = L::result(best);
        }
        continue;
      }
      // Border: the segment's own candidate list.
      for (int x = xs; x < xe; ++x) {
        Best best;
        for (int r = 0; r < n_rows; ++r) {
          const Cell* const row = rows[static_cast<std::size_t>(r)];
          for (int c = 0; c < n_cols; ++c) {
            layout.fold(best, y, x, row[x + offs[static_cast<std::size_t>(c)]]);
          }
        }
        dst[x] = L::result(best);
      }
    }
  }
}

// The Packed kernel's pass. Same segments as pass_rows_split, but no segment
// needs a candidate list: a candidate row that is out of bounds is replaced
// by `none`, a row of W absent sites (-1), which the fold skips exactly as
// the reference skips the out-of-bounds row; and the column offsets that a
// segment can use are template parameters. So every segment, interior or
// border, runs the same branch-free loop with 3 to 9 candidates, which
// Clang vectorizes when asked to (VORONAV_VECTORIZE_LOOP; its cost model
// declines on its own, see docs/CORE.md, H4).
template <bool kLeft, bool kRight>
VORONAV_NOINLINE void packed_segment(const int32_t* r0, const int32_t* r1, const int32_t* r2,
                                     int32_t* dst, int y, int k, int xs, int xe) noexcept {
  const auto pixel = [r0, r1, r2, y, k](int x) noexcept {
    Packed32::Best best;
    const auto row = [&best, y, k, x](const int32_t* r) noexcept {
      static_cast<void>(k);  // unused when neither side column is in bounds
      if constexpr (kLeft) {
        Packed32::fold(best, y, x, r[x - k]);
      }
      Packed32::fold(best, y, x, r[x]);
      if constexpr (kRight) {
        Packed32::fold(best, y, x, r[x + k]);
      }
    };
    row(r0);
    row(r1);
    row(r2);
    return Packed32::result(best);
  };
  VORONAV_VECTORIZE_LOOP
  for (int x = xs; x < xe; ++x) {
    dst[x] = pixel(x);
  }
}

VORONAV_NOINLINE void pass_rows_packed(const int32_t* cur, const int32_t* none, int32_t* out, int h,
                                       int w, int k, int y0, int y1) noexcept {
  const auto row_ptr = [w]<typename T>(T* base, int64_t y) noexcept {
    return base + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(w);
  };
  const int lo = std::clamp(std::min(k, w - k), 0, w);
  const int hi = std::clamp(std::max(k, w - k), 0, w);
  const std::array<int, 4> bounds{0, lo, hi, w};
  for (int y = y0; y < y1; ++y) {
    const int64_t up = int64_t{y} - k;  // int64: y + k cannot overflow
    const int64_t down = int64_t{y} + k;
    const int32_t* const r0 = up >= 0 ? row_ptr(cur, up) : none;
    const int32_t* const r1 = row_ptr(cur, y);
    const int32_t* const r2 = down < h ? row_ptr(cur, down) : none;
    int32_t* const dst = row_ptr(out, y);
    for (std::size_t seg = 0; seg < 3; ++seg) {
      const int xs = bounds[seg];
      const int xe = bounds[seg + 1];
      if (xs >= xe) {
        continue;
      }
      const bool left = xs >= k;               // x - k in bounds over the segment
      const bool right = int64_t{xs} + k < w;  // x + k in bounds over the segment
      if (left && right) {
        packed_segment<true, true>(r0, r1, r2, dst, y, k, xs, xe);
      } else if (left) {
        packed_segment<true, false>(r0, r1, r2, dst, y, k, xs, xe);
      } else if (right) {
        packed_segment<false, true>(r0, r1, r2, dst, y, k, xs, xe);
      } else {
        packed_segment<false, false>(r0, r1, r2, dst, y, k, xs, xe);
      }
    }
  }
}

// Runs fn(y0, y1) over all rows, on the pool or serially.
template <typename Fn>
void for_rows(int h, ThreadPool* pool, Schedule schedule, int chunk, Fn& fn) {
  if (pool == nullptr) {
    fn(0, h);
  } else {
    pool->parallel_rows(h, schedule, chunk, fn);  // returns after every row is written
  }
}

// The double-buffered pass loop shared by every kernel: `pass(src, dst, k,
// y0, y1)` computes rows [y0, y1) of one pass.
// kFill = false allocates the second buffer without zeroing it: the first
// pass writes every cell before anything reads it (H6; the Reference and
// Split kernels keep the C1-C3 allocation).
template <bool kFill, typename Cell, typename Pass>
Grid<Cell> run_passes(Grid<Cell> cur, std::span<const int> steps, ThreadPool* pool,
                      Schedule schedule, int chunk, const Pass& pass) {
  Grid<Cell> nxt = kFill ? Grid<Cell>(cur.height(), cur.width())
                         : Grid<Cell>(cur.height(), cur.width(), kUninitialized);
  const int h = cur.height();
  for (const int k : steps) {
    const GridView<const Cell> src = cur.view();
    const GridView<Cell> dst = nxt.view();
    auto rows = [&pass, src, dst, k](int y0, int y1) noexcept { pass(src, dst, k, y0, y1); };
    for_rows(h, pool, schedule, chunk, rows);
    std::swap(cur, nxt);  // moves keep the buffers, so views stay valid
  }
  return cur;
}

// Flat int64 -> packed for rows [y0, y1): one division per pixel, once per
// call (the reference kernel does 9 per pixel per pass).
VORONAV_NOINLINE void to_packed_rows(GridView<const int64_t> src, GridView<int32_t> dst, int y0,
                                     int y1) noexcept {
  const int64_t w = src.width;
  for (int y = y0; y < y1; ++y) {
    const int64_t* const s = src.row(y).data();
    int32_t* const d = dst.row(y).data();
    for (int64_t x = 0; x < w; ++x) {
      const int64_t v = s[x];
      const int64_t sy = v / w;
      d[x] = v < 0 ? -1 : static_cast<int32_t>((sy << 16) | (v - sy * w));
    }
  }
}

// Packed -> flat int64 for rows [y0, y1): a multiply-add per pixel.
VORONAV_NOINLINE void from_packed_rows(GridView<const int32_t> src, GridView<int64_t> dst, int y0,
                                       int y1) noexcept {
  const int64_t w = src.width;
  for (int y = y0; y < y1; ++y) {
    const int32_t* const s = src.row(y).data();
    int64_t* const d = dst.row(y).data();
    for (int64_t x = 0; x < w; ++x) {
      const int32_t p = s[x];
      d[x] = p < 0 ? kNoSite : static_cast<int64_t>(p >> 16) * w + static_cast<int64_t>(p & 0xFFFF);
    }
  }
}

Grid<int64_t> run_packed(GridView<const int64_t> init, std::span<const int> steps, ThreadPool* pool,
                         Schedule schedule, int chunk) {
  const int h = init.height;
  const int w = init.width;
  Grid<int32_t> packed(h, w, kUninitialized);  // every cell written below
  {
    auto rows = [init, dst = packed.view()](int y0, int y1) noexcept {
      to_packed_rows(init, dst, y0, y1);
    };
    for_rows(h, pool, schedule, chunk, rows);
  }
  const Grid<int32_t> none(1, w, -1);  // stands in for out-of-bounds candidate rows
  const auto pass = [h, w, &none](GridView<const int32_t> src, GridView<int32_t> dst, int k, int y0,
                                  int y1) noexcept {
    pass_rows_packed(src.data, none.data(), dst.data, h, w, k, y0, y1);
  };
  const Grid<int32_t> result =
      run_passes<false>(std::move(packed), steps, pool, schedule, chunk, pass);
  Grid<int64_t> out(h, w, kUninitialized);  // every cell written below
  {
    auto rows = [src = result.view(), dst = out.view()](int y0, int y1) noexcept {
      from_packed_rows(src, dst, y0, y1);
    };
    for_rows(h, pool, schedule, chunk, rows);
  }
  return out;
}

}  // namespace

// out[y, x] = the best of cur's 3x3 candidates at offsets {-k, 0, k}, by the
// (d2, site) tie-break documented in jfa.hpp, for y in [y0, y1). Mirrors
// _pixel in jfa_numba.py. All arithmetic is int64: with sides below 2^31,
// |dy|, |dx| < 2^31, so d2 < 2^63.
void jfa_pass_rows(GridView<const int64_t> cur, GridView<int64_t> out, int64_t k, int y0,
                   int y1) noexcept {
  const int64_t h = cur.height;
  const int64_t w = cur.width;
  for (int64_t y = y0; y < y1; ++y) {
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

Kernel resolve_kernel(int height, int width, Kernel kernel) noexcept {
  if (kernel != Kernel::Auto) {
    return kernel;
  }
  return height <= kPackedMaxSide && width <= kPackedMaxSide ? Kernel::Packed : Kernel::Split;
}

Grid<int64_t> run_steps(GridView<const int64_t> init, std::span<const int> steps, ThreadPool* pool,
                        Schedule schedule, int chunk, Kernel kernel) {
  if (chunk < 1) {
    throw std::invalid_argument("run_steps: chunk must be >= 1");
  }
  check_nearest(init, "run_steps", pool, schedule, chunk);
  const int h = init.height;
  const int w = init.width;
  switch (resolve_kernel(h, w, kernel)) {
    case Kernel::Reference: {
      const auto pass = [](GridView<const int64_t> src, GridView<int64_t> dst, int k, int y0,
                           int y1) noexcept { jfa_pass_rows(src, dst, k, y0, y1); };
      return run_passes<true>(to_grid(init), steps, pool, schedule, chunk, pass);
    }
    case Kernel::Split: {
      const auto pass = [h, w](GridView<const int64_t> src, GridView<int64_t> dst, int k, int y0,
                               int y1) noexcept {
        pass_rows_split(Flat64{w}, src.data, dst.data, h, w, k, y0, y1);
      };
      return run_passes<true>(to_grid(init), steps, pool, schedule, chunk, pass);
    }
    case Kernel::Packed:
      if (h > kPackedMaxSide || w > kPackedMaxSide) {
        throw std::invalid_argument("run_steps: Kernel::Packed needs both sides <= 32767");
      }
      return run_packed(init, steps, pool, schedule, chunk);
    default:
      throw std::invalid_argument("run_steps: unknown kernel");
  }
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
