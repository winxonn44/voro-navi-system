// Unit tests for jfa.hpp. They mirror tests/test_jfa_properties.py; inputs are
// built by hand or from a seeded std::mt19937_64, never from Python.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/oracle.hpp"

namespace voronav {
namespace {

using Steps = std::vector<int>;

Grid<int64_t> run(const Grid<uint8_t>& sites, Variant variant) {
  const Grid<int64_t> init = init_nearest(sites.view());
  const Steps steps = step_schedule(sites.height(), sites.width(), variant);
  return run_steps(init.view(), steps);
}

// Exactly n distinct sites at uniformly random cells, like maps.random_sites.
Grid<uint8_t> random_sites(int h, int w, int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::size_t> cells(static_cast<std::size_t>(h) * static_cast<std::size_t>(w));
  for (std::size_t i = 0; i < cells.size(); ++i) {
    cells[i] = i;
  }
  Grid<uint8_t> sites(h, w, 0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    std::uniform_int_distribution<std::size_t> pick(i, cells.size() - 1);
    std::swap(cells[i], cells[pick(rng)]);
    sites.data()[cells[i]] = 1;
  }
  return sites;
}

TEST(StepSchedule, MatchesPythonReference) {
  // The cases tests/test_jfa_properties.py::test_step_schedule pins.
  EXPECT_EQ(step_schedule(100, 60, Variant::Jfa), (Steps{64, 32, 16, 8, 4, 2, 1}));
  EXPECT_EQ(step_schedule(8, 8, Variant::JfaPlus1), (Steps{4, 2, 1, 1}));
  EXPECT_EQ(step_schedule(8, 8, Variant::JfaPlus2), (Steps{4, 2, 1, 2, 1}));
  EXPECT_EQ(step_schedule(8, 8, Variant::OnePlusJfa), (Steps{1, 4, 2, 1}));
  EXPECT_EQ(step_schedule(1, 1, Variant::Jfa), Steps{});
}

TEST(StepSchedule, OneByOneEveryVariant) {
  EXPECT_EQ(step_schedule(1, 1, Variant::Jfa), Steps{});
  EXPECT_EQ(step_schedule(1, 1, Variant::JfaPlus1), (Steps{1}));
  EXPECT_EQ(step_schedule(1, 1, Variant::JfaPlus2), (Steps{2, 1}));
  EXPECT_EQ(step_schedule(1, 1, Variant::OnePlusJfa), (Steps{1}));
  EXPECT_EQ(step_schedule(0, 0, Variant::Jfa), Steps{});
}

TEST(StepSchedule, OddAndNonPowerOfTwoSizes) {
  // N = max side rounded up to a power of two; the steps are N/2 ... 1.
  EXPECT_EQ(step_schedule(1, 2, Variant::Jfa), (Steps{1}));
  EXPECT_EQ(step_schedule(2, 3, Variant::Jfa), (Steps{2, 1}));
  EXPECT_EQ(step_schedule(1, 17, Variant::Jfa), (Steps{16, 8, 4, 2, 1}));
  EXPECT_EQ(step_schedule(23, 1, Variant::Jfa), (Steps{16, 8, 4, 2, 1}));
  EXPECT_EQ(step_schedule(5, 5, Variant::JfaPlus2), (Steps{4, 2, 1, 2, 1}));
  EXPECT_EQ(step_schedule(16, 16, Variant::Jfa), (Steps{8, 4, 2, 1}));
  EXPECT_EQ(step_schedule(17, 3, Variant::OnePlusJfa), (Steps{1, 16, 8, 4, 2, 1}));
  EXPECT_EQ(step_schedule(61, 89, Variant::JfaPlus1), (Steps{64, 32, 16, 8, 4, 2, 1, 1}));
}

TEST(StepSchedule, HugeSidesDoNotOverflow) {
  // INT_MAX rounds up to 2^31, which does not fit in int; the first step does.
  const Steps steps = step_schedule(INT_MAX, 1, Variant::Jfa);
  ASSERT_EQ(steps.size(), 31U);
  EXPECT_EQ(steps.front(), 1 << 30);
  EXPECT_EQ(steps.back(), 1);
}

TEST(StepSchedule, RejectsUnknownVariant) {
  EXPECT_THROW(static_cast<void>(step_schedule(8, 8, static_cast<Variant>(7))),
               std::invalid_argument);
}

TEST(InitNearest, SitesPointAtThemselves) {
  Grid<uint8_t> sites(3, 4, 0);
  sites(0, 1) = 1;
  sites(2, 3) = 1;
  const Grid<int64_t> init = init_nearest(sites.view());
  ASSERT_EQ(init.height(), 3);
  ASSERT_EQ(init.width(), 4);
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 4; ++x) {
      const int64_t expected = sites(y, x) != 0 ? int64_t{y} * 4 + x : kNoSite;
      EXPECT_EQ(init(y, x), expected) << y << "," << x;
    }
  }
}

TEST(DistanceSq, HandComputed) {
  Grid<int64_t> nearest(2, 3, kNoSite);
  nearest(0, 0) = 5;  // site at (1, 2)
  nearest(0, 2) = 5;
  nearest(1, 2) = 5;
  nearest(1, 0) = 0;  // site at (0, 0)
  const Grid<int64_t> d2 = distance_sq(nearest.view());
  EXPECT_EQ(d2(0, 0), 5);  // dy = -1, dx = -2
  EXPECT_EQ(d2(0, 1), -1);
  EXPECT_EQ(d2(0, 2), 1);
  EXPECT_EQ(d2(1, 0), 1);
  EXPECT_EQ(d2(1, 1), -1);
  EXPECT_EQ(d2(1, 2), 0);
}

TEST(DistanceSq, WideGridNeedsInt64) {
  // 59,999^2 is about 3.6e9, past INT32_MAX: d2 must be computed in int64.
  constexpr int kW = 60000;
  Grid<int64_t> nearest(1, kW, 0);
  const Grid<int64_t> d2 = distance_sq(nearest.view());
  EXPECT_EQ(d2(0, kW - 1), int64_t{kW - 1} * (kW - 1));
}

TEST(DistanceSq, RejectsInvalidEntries) {
  Grid<int64_t> nearest(2, 2, kNoSite);
  nearest(1, 1) = 4;  // == H * W, one past the end
  EXPECT_THROW(static_cast<void>(distance_sq(nearest.view())), std::invalid_argument);
  nearest(1, 1) = -2;
  EXPECT_THROW(static_cast<void>(distance_sq(nearest.view())), std::invalid_argument);
}

TEST(RunSteps, RejectsInvalidInit) {
  Grid<int64_t> init(3, 3, kNoSite);
  init(0, 0) = 9;
  const Steps steps{1};
  EXPECT_THROW(static_cast<void>(run_steps(init.view(), steps)), std::invalid_argument);
}

TEST(RunSteps, NoStepsReturnsACopy) {
  Grid<int64_t> init(2, 2, kNoSite);
  init(1, 0) = 2;
  const Grid<int64_t> out = run_steps(init.view(), std::span<const int>{});
  EXPECT_EQ(out, init);
  EXPECT_NE(out.data(), init.data());
}

TEST(RunSteps, EmptyGrids) {
  for (const auto& [h, w] : {std::pair{0, 0}, std::pair{0, 5}, std::pair{4, 0}}) {
    const Grid<uint8_t> sites(h, w, 0);
    const Grid<int64_t> out = run(sites, Variant::JfaPlus2);
    EXPECT_EQ(out.height(), h);
    EXPECT_EQ(out.width(), w);
    EXPECT_EQ(out.size(), 0U);
  }
}

TEST(RunSteps, NoSitesLeavesEverythingUnassigned) {
  const Grid<uint8_t> sites(9, 13, 0);
  for (const Variant v :
       {Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2, Variant::OnePlusJfa}) {
    EXPECT_EQ(run(sites, v), Grid<int64_t>(9, 13, kNoSite));
  }
}

TEST(RunSteps, SingleSiteIsExact) {
  // With one site the halving schedule reaches every pixel with the only
  // candidate, so plain JFA is exact (test_single_site_is_exact).
  Grid<uint8_t> sites(37, 53, 0);
  sites(11, 40) = 1;
  EXPECT_EQ(run(sites, Variant::Jfa), Grid<int64_t>(37, 53, int64_t{11} * 53 + 40));
}

TEST(RunSteps, SingleSiteOnAWideGrid) {
  constexpr int kW = 60000;
  Grid<uint8_t> sites(1, kW, 0);
  sites(0, kW - 1) = 1;
  EXPECT_EQ(run(sites, Variant::Jfa), Grid<int64_t>(1, kW, kW - 1));
}

TEST(RunSteps, AllSitesMapToThemselves) {
  const Grid<uint8_t> sites(7, 11, 1);
  for (const Variant v :
       {Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2, Variant::OnePlusJfa}) {
    EXPECT_EQ(run(sites, v), init_nearest(sites.view()));
  }
}

TEST(RunSteps, DegenerateShapesMatchOracle) {
  for (const auto& [h, w] :
       {std::pair{1, 1}, std::pair{1, 17}, std::pair{23, 1}, std::pair{2, 3}}) {
    Grid<uint8_t> sites(h, w, 0);
    sites.data()[sites.size() - 1] = 1;
    EXPECT_EQ(run(sites, Variant::JfaPlus1), brute_force(sites.view())) << h << "x" << w;
  }
}

TEST(TieBreak, LowerFlatIndexWinsInJfaAndOracle) {
  // Pixel 2 is equidistant (d2 = 4) from sites 0 and 4.
  Grid<uint8_t> sites(1, 5, 0);
  sites(0, 0) = 1;
  sites(0, 4) = 1;
  const Grid<int64_t> expected = [] {
    Grid<int64_t> g(1, 5);
    const std::array<int64_t, 5> row{0, 0, 0, 4, 4};
    std::ranges::copy(row, g.data());
    return g;
  }();
  EXPECT_EQ(brute_force(sites.view()), expected);
  for (const Variant v :
       {Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2, Variant::OnePlusJfa}) {
    EXPECT_EQ(run(sites, v), expected);
  }

  // A 2D tie: (2, 2) is at d2 = 4 from site (0, 2) = 2 and site (2, 0) = 10.
  Grid<uint8_t> sites2(5, 5, 0);
  sites2(0, 2) = 1;
  sites2(2, 0) = 1;
  EXPECT_EQ(brute_force(sites2.view())(2, 2), 2);
  EXPECT_EQ(run(sites2, Variant::JfaPlus2)(2, 2), 2);
}

TEST(TieBreak, IndependentOfCandidateVisitOrder) {
  // Candidates are visited left to right, so put the higher-index site on the
  // left: pixel 1 sees site 2 (via x = 0) before site 0 (via x = 2), both at
  // d2 = 1. The lower index must still win, as the packed key would choose.
  Grid<int64_t> init(1, 3, kNoSite);
  init(0, 0) = 2;
  init(0, 2) = 0;
  const Steps one{1};
  EXPECT_EQ(run_steps(init.view(), one)(0, 1), 0);
}

TEST(RunSteps, CoreInvariantsEveryVariant) {
  // test_core_invariants: 61 x 89 with 45 sites.
  const Grid<uint8_t> sites = random_sites(61, 89, 45, 11);
  const Grid<int64_t> ref_d2 = distance_sq(brute_force(sites.view()).view());
  for (const Variant v :
       {Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2, Variant::OnePlusJfa}) {
    const Grid<int64_t> out = run(sites, v);
    const Grid<int64_t> d2 = distance_sq(out.view());
    std::size_t exact = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
      const int64_t s = out.data()[i];
      ASSERT_GE(s, 0);                                          // every pixel assigned
      ASSERT_NE(sites.data()[static_cast<std::size_t>(s)], 0);  // to a real site
      if (sites.data()[i] != 0) {
        ASSERT_EQ(s, static_cast<int64_t>(i));  // sites are their own
      }
      ASSERT_GE(d2.data()[i], ref_d2.data()[i]);  // never too close
      exact += d2.data()[i] == ref_d2.data()[i] ? 1U : 0U;
    }
    EXPECT_GT(exact, 0U);
  }
}

TEST(RunSteps, DoubleBufferingKeepsInputIntact) {
  const Grid<uint8_t> sites = random_sites(20, 30, 12, 3);
  const Grid<int64_t> init = init_nearest(sites.view());
  const Grid<int64_t> before = init;
  const Steps steps = step_schedule(20, 30, Variant::JfaPlus2);
  static_cast<void>(run_steps(init.view(), steps));
  EXPECT_EQ(init, before);
}

}  // namespace
}  // namespace voronav
