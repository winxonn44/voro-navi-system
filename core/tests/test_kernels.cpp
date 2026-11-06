// The C4 kernels (jfa.hpp Kernel::Split, Kernel::Packed, Kernel::Auto) against
// the C1 reference kernel: exact Grid equality, serial and on pools, on shapes
// chosen for the interior/border split (no interior at all, interior only a
// row or a column wide, segments that overlap when 2k > W) and for the packed
// int16 layout (at its size limit, and the fallback just above it). Grids
// come from test_random.hpp, so every platform checks the same inputs.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

#include "test_random.hpp"

namespace voronav {
namespace {

constexpr std::array<Variant, 4> kVariants{Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2,
                                           Variant::OnePlusJfa};
constexpr std::array<Kernel, 3> kFastKernels{Kernel::Split, Kernel::Packed, Kernel::Auto};

const char* name(Kernel k) {
  switch (k) {
    case Kernel::Auto:
      return "auto";
    case Kernel::Reference:
      return "reference";
    case Kernel::Split:
      return "split";
    case Kernel::Packed:
      return "packed";
  }
  return "?";
}

Grid<uint8_t> random_grid(int h, int w, double density, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Grid<uint8_t> sites(h, w, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = test::bernoulli(rng, density) ? 1 : 0;
  }
  return sites;
}

// Every fast kernel, serially, gives the reference kernel's grid.
void expect_kernels_match(const Grid<uint8_t>& sites, const std::vector<int>& steps) {
  const Grid<int64_t> init = init_nearest(sites.view());
  const Grid<int64_t> ref =
      run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, Kernel::Reference);
  for (const Kernel k : kFastKernels) {
    const Grid<int64_t> out =
        run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, k);
    EXPECT_EQ(out, ref) << sites.height() << "x" << sites.width() << " kernel " << name(k);
  }
}

struct Shape {
  int h;
  int w;
};

TEST(Kernels, MatchReferenceOnBorderHeavyShapes) {
  // Sides 0-3 have no interior at any step; 4-6 only at k = 1; 1xN / Nx1
  // never have 3 rows (or columns) in bounds; 2 x N has border rows only;
  // and widths just below / at / above powers of two move the breakpoints
  // k and W - k past each other.
  std::vector<Shape> shapes;
  for (int h = 0; h <= 6; ++h) {
    for (int w = 0; w <= 6; ++w) {
      shapes.push_back({h, w});
    }
  }
  for (const Shape s : {Shape{1, 100}, Shape{100, 1}, Shape{2, 77}, Shape{77, 2}, Shape{3, 64},
                        Shape{64, 3}, Shape{7, 15}, Shape{15, 16}, Shape{16, 17}, Shape{31, 32},
                        Shape{33, 65}, Shape{65, 33}, Shape{12, 13}, Shape{128, 129}}) {
    shapes.push_back(s);
  }
  uint64_t seed = 0;
  int grids = 0;
  for (const Shape sh : shapes) {
    for (const double d : {0.0, 0.03, 0.2, 1.0}) {
      const Grid<uint8_t> sites = random_grid(sh.h, sh.w, d, seed++);
      for (const Variant v : kVariants) {
        expect_kernels_match(sites, step_schedule(sh.h, sh.w, v));
        ++grids;
      }
    }
  }
  RecordProperty("grids", grids);
}

TEST(Kernels, MatchReferenceOnRandomShapes) {
  std::mt19937_64 rng(42);
  for (int i = 0; i < 200; ++i) {
    const int h = test::uniform_int(rng, 1, 90);
    const int w = test::uniform_int(rng, 1, 90);
    const double d = std::array<double, 4>{0.002, 0.01, 0.05, 0.3}[static_cast<std::size_t>(i % 4)];
    const Grid<uint8_t> sites = random_grid(h, w, d, rng());
    expect_kernels_match(sites, step_schedule(h, w, kVariants[static_cast<std::size_t>(i % 4)]));
  }
}

TEST(Kernels, ArbitraryStepsIncludingStepsLargerThanTheGrid) {
  // run_steps accepts any step list; steps >= a side leave only the
  // centre candidate (and its row or column) in bounds.
  const Grid<uint8_t> sites = random_grid(20, 37, 0.05, 9);
  expect_kernels_match(sites, {100, 37, 36, 20, 19, 18, 10, 3, 1, 1, 2, 1});
}

TEST(Kernels, LatticeTiesPickTheLowerIndex) {
  // Sites on a regular lattice make many pixels equidistant from 2 or 4
  // sites; the packed comparison must break those ties exactly like the
  // flat index does.
  Grid<uint8_t> sites(33, 41, 0);
  for (int y = 0; y < 33; y += 4) {
    for (int x = 0; x < 41; x += 4) {
      sites(y, x) = 1;
    }
  }
  for (const Variant v : kVariants) {
    expect_kernels_match(sites, step_schedule(33, 41, v));
  }
}

TEST(Kernels, PinnedJfaPlus2Counterexample) {
  // C1's 12 x 13 grid: every kernel keeps the reference's wrong pixel 12.
  Grid<uint8_t> sites(12, 13, 0);
  sites.data()[39] = 1;
  sites.data()[151] = 1;
  const Grid<int64_t> init = init_nearest(sites.view());
  const std::vector<int> steps = step_schedule(12, 13, Variant::JfaPlus2);
  for (const Kernel k : kFastKernels) {
    const Grid<int64_t> out =
        run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, k);
    EXPECT_EQ(out.data()[12], 39) << name(k);
  }
  for (const Variant v : kVariants) {
    expect_kernels_match(sites, step_schedule(12, 13, v));
  }
}

TEST(Kernels, IdenticalOnPoolsForEveryScheduleAndChunk) {
  const std::array<Shape, 6> shapes{{{1, 50}, {3, 3}, {5, 200}, {97, 131}, {200, 23}, {256, 256}}};
  std::vector<std::unique_ptr<ThreadPool>> pools;
  for (const unsigned n : {2U, 3U, 10U}) {
    pools.push_back(std::make_unique<ThreadPool>(n));
  }
  uint64_t seed = 500;
  for (const Shape sh : shapes) {
    const Grid<uint8_t> sites = random_grid(sh.h, sh.w, 0.01, seed++);
    const Grid<int64_t> init = init_nearest(sites.view());
    const std::vector<int> steps = step_schedule(sh.h, sh.w, Variant::JfaPlus2);
    const Grid<int64_t> ref =
        run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, Kernel::Reference);
    for (const auto& pool : pools) {
      for (const Schedule s : {Schedule::Static, Schedule::Dynamic}) {
        for (const int chunk : {1, 4, 16}) {
          for (const Kernel k : {Kernel::Reference, Kernel::Split, Kernel::Packed}) {
            EXPECT_EQ(run_steps(init.view(), steps, pool.get(), s, chunk, k), ref)
                << sh.h << "x" << sh.w << " " << name(k) << " threads " << pool->size() << " chunk "
                << chunk;
          }
        }
      }
    }
  }
}

TEST(Kernels, ResolveKernel) {
  EXPECT_EQ(resolve_kernel(0, 0, Kernel::Auto), Kernel::Packed);
  EXPECT_EQ(resolve_kernel(2048, 2048, Kernel::Auto), Kernel::Packed);
  EXPECT_EQ(resolve_kernel(kPackedMaxSide, kPackedMaxSide, Kernel::Auto), Kernel::Packed);
  EXPECT_EQ(resolve_kernel(kPackedMaxSide + 1, 1, Kernel::Auto), Kernel::Split);
  EXPECT_EQ(resolve_kernel(1, kPackedMaxSide + 1, Kernel::Auto), Kernel::Split);
  for (const Kernel k : {Kernel::Reference, Kernel::Split, Kernel::Packed}) {
    EXPECT_EQ(resolve_kernel(1, kPackedMaxSide + 1, k), k);
  }
}

// Sites at both ends of the long side (so the largest distances the shape
// allows are computed), plus random ones.
Grid<uint8_t> long_grid(int h, int w, uint64_t seed) {
  Grid<uint8_t> sites = random_grid(h, w, 0.0005, seed);
  sites.data()[0] = 1;
  sites.data()[sites.size() - 1] = 1;
  return sites;
}

TEST(Kernels, PackedAtItsSizeLimit) {
  // Sides of exactly 32767: the largest packed coordinate (32766) in each
  // field, and d2 up to 32766^2. (The two-dimensional worst case,
  // 2 * 32766^2, needs a 32767^2 grid and is covered by a static_assert in
  // jfa.cpp instead.)
  for (const Shape sh : {Shape{1, kPackedMaxSide}, Shape{kPackedMaxSide, 1},
                         Shape{2, kPackedMaxSide}, Shape{kPackedMaxSide, 3}}) {
    const Grid<uint8_t> sites = long_grid(sh.h, sh.w, 77);
    for (const Variant v : {Variant::Jfa, Variant::JfaPlus2}) {
      expect_kernels_match(sites, step_schedule(sh.h, sh.w, v));
    }
  }
  // A single site at the far end: every pixel's d2 is the largest possible.
  Grid<uint8_t> one(1, kPackedMaxSide, 0);
  one.data()[one.size() - 1] = 1;
  const Grid<int64_t> init = init_nearest(one.view());
  const Grid<int64_t> out = run_steps(init.view(), step_schedule(1, kPackedMaxSide, Variant::Jfa),
                                      nullptr, Schedule::Dynamic, kDefaultChunk, Kernel::Packed);
  for (const int64_t s : out.view().flat()) {
    ASSERT_EQ(s, kPackedMaxSide - 1);
  }
}

TEST(Kernels, AutoFallsBackAboveTheLimitAndPackedRefuses) {
  for (const Shape sh :
       {Shape{1, kPackedMaxSide + 1}, Shape{kPackedMaxSide + 1, 1}, Shape{2, kPackedMaxSide + 5}}) {
    const Grid<uint8_t> sites = long_grid(sh.h, sh.w, 78);
    const Grid<int64_t> init = init_nearest(sites.view());
    const std::vector<int> steps = step_schedule(sh.h, sh.w, Variant::JfaPlus2);
    EXPECT_EQ(resolve_kernel(sh.h, sh.w, Kernel::Auto), Kernel::Split);
    const Grid<int64_t> ref =
        run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, Kernel::Reference);
    EXPECT_EQ(run_steps(init.view(), steps), ref);  // Auto
    ThreadPool pool(4);
    EXPECT_EQ(run_steps(init.view(), steps, &pool), ref);
    EXPECT_THROW(static_cast<void>(run_steps(init.view(), steps, nullptr, Schedule::Dynamic,
                                             kDefaultChunk, Kernel::Packed)),
                 std::invalid_argument);
  }
}

TEST(Kernels, ValidationAppliesToEveryKernel) {
  const Grid<int64_t> bad(4, 4, 16);  // == H*W: out of range
  const std::vector<int> steps{2, 1};
  for (const Kernel k : {Kernel::Auto, Kernel::Reference, Kernel::Split, Kernel::Packed}) {
    EXPECT_THROW(static_cast<void>(
                     run_steps(bad.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, k)),
                 std::invalid_argument)
        << name(k);
  }
  const Grid<int64_t> ok(4, 4, kNoSite);
  EXPECT_THROW(static_cast<void>(run_steps(ok.view(), steps, nullptr, Schedule::Dynamic,
                                           kDefaultChunk, static_cast<Kernel>(99))),
               std::invalid_argument);
}

TEST(Kernels, NoStepsRoundTripsThroughThePackedLayout) {
  const Grid<uint8_t> sites = random_grid(37, 300, 0.1, 3);
  const Grid<int64_t> init = init_nearest(sites.view());
  for (const Kernel k : kFastKernels) {
    EXPECT_EQ(run_steps(init.view(), {}, nullptr, Schedule::Dynamic, kDefaultChunk, k), init)
        << name(k);
  }
}

TEST(Kernels, WorkerQosDoesNotChangeResults) {
  ThreadPool pool(4, WorkerQos::UserInteractive);
  const Grid<uint8_t> sites = random_grid(150, 170, 0.01, 11);
  const Grid<int64_t> init = init_nearest(sites.view());
  const std::vector<int> steps = step_schedule(150, 170, Variant::JfaPlus2);
  const Grid<int64_t> ref =
      run_steps(init.view(), steps, nullptr, Schedule::Dynamic, kDefaultChunk, Kernel::Reference);
  for (const Schedule s : {Schedule::Static, Schedule::Dynamic}) {
    EXPECT_EQ(run_steps(init.view(), steps, &pool, s), ref);
  }
}

TEST(Kernels, PooledValidationFindsABadEntryAnywhere) {
  // H6 checks the input rows in parallel; a bad entry in any row, with any
  // schedule or chunk, must still throw, and a good grid must not.
  ThreadPool pool(5);
  for (const int64_t bad_value : {int64_t{-2}, int64_t{37} * 41}) {
    for (const std::size_t at : {std::size_t{0}, std::size_t{700}, std::size_t{37 * 41 - 1}}) {
      Grid<int64_t> init(37, 41, kNoSite);
      init.data()[at] = bad_value;
      for (const Kernel k : {Kernel::Auto, Kernel::Reference, Kernel::Split, Kernel::Packed}) {
        for (const Schedule s : {Schedule::Static, Schedule::Dynamic}) {
          for (const int chunk : {1, 16}) {
            EXPECT_THROW(static_cast<void>(
                             run_steps(init.view(), std::vector<int>{2, 1}, &pool, s, chunk, k)),
                         std::invalid_argument)
                << at << " " << name(k);
          }
        }
      }
    }
  }
  const Grid<int64_t> ok(37, 41, kNoSite);
  EXPECT_EQ(run_steps(ok.view(), std::vector<int>{2, 1}, &pool), ok);
}

TEST(Kernels, UninitializedGridHasTheShapeAndIsWritable) {
  Grid<int32_t> g(3, 5, kUninitialized);
  EXPECT_EQ(g.height(), 3);
  EXPECT_EQ(g.width(), 5);
  ASSERT_EQ(g.size(), 15U);
  for (std::size_t i = 0; i < g.size(); ++i) {
    g.data()[i] = static_cast<int32_t>(i);
  }
  EXPECT_EQ(g(2, 4), 14);
  EXPECT_EQ(Grid<int64_t>(0, 7, kUninitialized).size(), 0U);
  EXPECT_THROW(Grid<int32_t>(-1, 2, kUninitialized), std::invalid_argument);
  // The filling constructor still fills with the new allocator.
  const Grid<int64_t> filled(4, 4, 7);
  for (const int64_t v : filled.view().flat()) {
    EXPECT_EQ(v, 7);
  }
}

}  // namespace
}  // namespace voronav
