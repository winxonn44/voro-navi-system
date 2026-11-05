// Determinism of the parallel JFA (design doc A5): run_steps gives exactly
// the serial result, Grid for Grid, for every thread count, schedule and
// chunk size, including shapes with fewer rows than threads. Grids come from
// test_random.hpp so every platform checks the same inputs. CI runs this
// under TSan, which would flag a pass that reads its own output or a missing
// barrier between passes.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

#include "test_random.hpp"

namespace voronav {
namespace {

constexpr std::array<Variant, 4> kVariants{Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2,
                                           Variant::OnePlusJfa};
constexpr std::array<Schedule, 2> kSchedules{Schedule::Static, Schedule::Dynamic};
constexpr std::array<int, 3> kChunks{1, 4, 16};

// {1, 2, 7, 10}, plus this machine's hardware concurrency if different.
std::vector<unsigned> thread_counts() {
  std::vector<unsigned> counts{1, 2, 7, 10};
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw != 0 && hw != 1 && hw != 2 && hw != 7 && hw != 10) {
    counts.push_back(hw);
  }
  return counts;
}

// One pool per thread count, created once and reused by every test here, as
// the Python module does.
const std::vector<std::unique_ptr<ThreadPool>>& pools() {
  static const auto* const all = [] {
    auto* p = new std::vector<std::unique_ptr<ThreadPool>>;
    for (const unsigned n : thread_counts()) {
      p->push_back(std::make_unique<ThreadPool>(n));
    }
    return p;
  }();
  return *all;
}

Grid<uint8_t> random_grid(int h, int w, double density, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Grid<uint8_t> sites(h, w, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = test::bernoulli(rng, density) ? 1 : 0;
  }
  return sites;
}

// Runs every (pool, schedule, chunk) on one seed grid and expects the serial
// result each time. Returns the number of parallel runs compared.
int expect_all_match(const Grid<uint8_t>& sites, Variant variant) {
  const Grid<int64_t> init = init_nearest(sites.view());
  const std::vector<int> steps = step_schedule(sites.height(), sites.width(), variant);
  const Grid<int64_t> serial = run_steps(init.view(), steps);
  int runs = 0;
  for (const auto& pool : pools()) {
    for (const Schedule s : kSchedules) {
      for (const int chunk : kChunks) {
        const Grid<int64_t> par = run_steps(init.view(), steps, pool.get(), s, chunk);
        EXPECT_EQ(par, serial) << sites.height() << "x" << sites.width() << " variant "
                               << static_cast<int>(variant) << " threads " << pool->size()
                               << (s == Schedule::Static ? " static" : " dynamic") << " chunk "
                               << chunk;
        ++runs;
      }
    }
  }
  return runs;
}

struct Shape {
  int h;
  int w;
};

TEST(ParallelDeterminism, IdenticalForEveryThreadCountScheduleAndChunk) {
  // Degenerate (empty, 1x1, 1xN, Nx1), fewer rows than threads (2-9 rows),
  // odd, non-square, both sides of a power of two, and a few hundred rows.
  const std::array<Shape, 16> shapes{{{0, 0},
                                      {0, 7},
                                      {5, 0},
                                      {1, 1},
                                      {1, 37},
                                      {41, 1},
                                      {2, 2},
                                      {3, 50},
                                      {9, 9},
                                      {12, 13},
                                      {17, 15},
                                      {31, 33},
                                      {63, 65},
                                      {64, 64},
                                      {97, 131},
                                      {200, 23}}};
  const std::array<double, 4> densities{0.0, 0.01, 0.1, 1.0};
  int grids = 0;
  int runs = 0;
  uint64_t seed = 0;
  for (const Shape& sh : shapes) {
    for (const double d : densities) {
      const Grid<uint8_t> sites = random_grid(sh.h, sh.w, d, seed++);
      for (const Variant v : kVariants) {
        runs += expect_all_match(sites, v);
        ++grids;
      }
    }
  }
  const auto configs = static_cast<int>(pools().size() * kSchedules.size() * kChunks.size());
  EXPECT_EQ(runs, grids * configs);
  RecordProperty("grids", grids);
  RecordProperty("parallel_runs", runs);
}

TEST(ParallelDeterminism, LargerGridsUnderContention) {
  // Enough rows (and passes) that every worker claims many chunks.
  int seed = 100;
  for (const Shape sh : {Shape{256, 256}, Shape{301, 177}}) {
    for (const double d : {0.001, 0.02}) {
      expect_all_match(random_grid(sh.h, sh.w, d, static_cast<uint64_t>(seed++)),
                       Variant::JfaPlus2);
    }
  }
}

TEST(ParallelDeterminism, PinnedJfaPlus2CounterexampleIsWrongIdentically) {
  // C1's minimal jfa+2 counterexample (test_oracle.cpp): 12 x 13, sites 39
  // and 151. Every parallel run must reproduce the serial output, including
  // its one wrong pixel (12 keeps site 39 instead of 151).
  Grid<uint8_t> sites(12, 13, 0);
  sites.data()[39] = 1;
  sites.data()[151] = 1;
  for (const Variant v : kVariants) {
    expect_all_match(sites, v);
  }
  const Grid<int64_t> init = init_nearest(sites.view());
  const std::vector<int> steps = step_schedule(12, 13, Variant::JfaPlus2);
  for (const auto& pool : pools()) {
    const Grid<int64_t> out = run_steps(init.view(), steps, pool.get(), Schedule::Dynamic, 1);
    EXPECT_EQ(out.data()[12], 39);
  }
}

TEST(ParallelRunSteps, ValidatesLikeSerial) {
  ThreadPool pool(3);
  const Grid<int64_t> bad(4, 4, 99);
  const std::vector<int> steps{2, 1};
  EXPECT_THROW(static_cast<void>(run_steps(bad.view(), steps, &pool)), std::invalid_argument);
  const Grid<int64_t> ok(4, 4, kNoSite);
  EXPECT_THROW(static_cast<void>(run_steps(ok.view(), steps, &pool, Schedule::Dynamic, 0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(run_steps(ok.view(), steps, nullptr, Schedule::Dynamic, 0)),
               std::invalid_argument);
}

TEST(ParallelRunSteps, KeepsInputIntact) {
  ThreadPool pool(4);
  const Grid<uint8_t> sites = random_grid(40, 50, 0.02, 7);
  const Grid<int64_t> init = init_nearest(sites.view());
  const Grid<int64_t> before = to_grid(init.view());
  const std::vector<int> steps = step_schedule(40, 50, Variant::JfaPlus2);
  static_cast<void>(run_steps(init.view(), steps, &pool, Schedule::Static));
  EXPECT_EQ(init, before);
}

}  // namespace
}  // namespace voronav
