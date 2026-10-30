// The brute-force oracle, and JFA checked against it on 1,000 random grids.
// Mirrors tests/test_oracle.py and the oracle properties of
// tests/test_jfa_properties.py; inputs come from a seeded std::mt19937_64.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/oracle.hpp"

#include "test_random.hpp"

namespace voronav {
namespace {

TEST(Oracle, TieBreakPrefersSmallerFlatIndex) {
  // Pixel 2 is equidistant from site 0 and site 4, so site 0 wins.
  Grid<uint8_t> sites(1, 5, 0);
  sites(0, 0) = 1;
  sites(0, 4) = 1;
  const Grid<int64_t> out = brute_force(sites.view());
  const std::array<int64_t, 5> expected{0, 0, 0, 4, 4};
  for (int x = 0; x < 5; ++x) {
    EXPECT_EQ(out(0, x), expected[static_cast<std::size_t>(x)]) << x;
  }
}

TEST(Oracle, NoSites) {
  EXPECT_EQ(brute_force(Grid<uint8_t>(4, 6, 0).view()), Grid<int64_t>(4, 6, kNoSite));
  EXPECT_EQ(brute_force(Grid<uint8_t>(0, 3, 0).view()).size(), 0U);
}

TEST(Oracle, HandComputedDistances) {
  // One site in the middle of 3 x 3, and a second one at the corner (2, 2):
  // pixel (1, 2) is then at d2 = 1 from both, and the centre (index 4) wins.
  Grid<uint8_t> sites(3, 3, 0);
  sites(1, 1) = 1;
  const Grid<int64_t> d2 = distance_sq(brute_force(sites.view()).view());
  const std::array<int64_t, 9> expected{2, 1, 2, 1, 0, 1, 2, 1, 2};
  for (std::size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(d2.data()[i], expected[i]) << i;
  }
  sites(2, 2) = 1;
  const Grid<int64_t> two = brute_force(sites.view());
  EXPECT_EQ(two(1, 2), 4);
  EXPECT_EQ(two(2, 1), 4);
  EXPECT_EQ(two(2, 2), 8);
}

TEST(Oracle, AssignsRealSitesAndSitesToThemselves) {
  std::mt19937_64 rng(5);
  Grid<uint8_t> sites(30, 30, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = static_cast<uint8_t>(test::bernoulli(rng, 0.03));
  }
  sites(0, 0) = 1;  // at least one
  const Grid<int64_t> out = brute_force(sites.view());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const int64_t s = out.data()[i];
    ASSERT_GE(s, 0);
    EXPECT_NE(sites.data()[static_cast<std::size_t>(s)], 0);
    if (sites.data()[i] != 0) {
      EXPECT_EQ(s, static_cast<int64_t>(i));
    }
  }
}

TEST(TestRandom, DrawsArePinned) {
  // The oracle-agreement grids must be the same on every platform. The
  // standard pins mt19937_64's 10,000th output; the helpers' outputs are
  // pinned here, so a platform that drew different grids fails this first.
  std::mt19937_64 engine;
  engine.discard(9999);
  EXPECT_EQ(engine(), 9981545732273789042ULL);

  std::mt19937_64 rng(42);
  const std::array<uint64_t, 4> below{934706, 854435, 539344, 822067};
  for (const uint64_t expected : below) {
    EXPECT_EQ(test::uniform_below(rng, 1000003), expected);
  }
  const std::array<int, 4> ints{0, 21, 2, 1};
  for (const int expected : ints) {
    EXPECT_EQ(test::uniform_int(rng, -5, 27), expected);
  }
  int heads = 0;
  for (int i = 0; i < 1000; ++i) {
    heads += test::bernoulli(rng, 0.3) ? 1 : 0;
  }
  EXPECT_EQ(heads, 291);
}

// --- JFA vs the oracle on random grids -------------------------------------

constexpr int kGrids = 1000;
constexpr std::array<double, 5> kDensities{0.002, 0.01, 0.05, 0.2, 0.6};
constexpr std::array<Variant, 4> kVariants{Variant::Jfa, Variant::JfaPlus1, Variant::JfaPlus2,
                                           Variant::OnePlusJfa};
constexpr std::array<const char*, 4> kVariantNames{"jfa", "jfa+1", "jfa+2", "1+jfa"};

// Grid i: density kDensities[i % 5]; sides uniform in [1, 64], so most grids
// are non-square and many are odd or 1 wide. Every 25th grid at the two lowest
// densities is 96-128 per side instead, for longer schedules. Drawn with the
// portable helpers of test_random.hpp, so every platform tests the same grids.
Grid<uint8_t> random_grid(int i) {
  std::mt19937_64 rng(0x5eed0000ULL + static_cast<uint64_t>(i));
  const bool large = i % 25 < 2;
  const int h = test::uniform_int(rng, large ? 96 : 1, large ? 128 : 64);
  const int w = test::uniform_int(rng, large ? 96 : 1, large ? 128 : 64);
  const double density = kDensities[static_cast<std::size_t>(i) % kDensities.size()];
  Grid<uint8_t> sites(h, w, 0);
  for (std::size_t p = 0; p < sites.size(); ++p) {
    sites.data()[p] = static_cast<uint8_t>(test::bernoulli(rng, density));
  }
  return sites;
}

TEST(OracleAgreement, ThousandRandomGrids) {
  // Per variant: pixels whose distance differs from the oracle's, and
  // violations of the invariants every variant must keep.
  std::array<std::array<int64_t, kDensities.size()>, kVariants.size()> wrong{};
  std::array<int64_t, kDensities.size()> pixels{};
  std::array<int64_t, kVariants.size()> invariant_failures{};
  int64_t label_differs_jfa2 = 0;  // equal distance, different (tied) site
  int first_bad_jfa2 = -1;

  for (int i = 0; i < kGrids; ++i) {
    const Grid<uint8_t> sites = random_grid(i);
    const std::size_t di = static_cast<std::size_t>(i) % kDensities.size();
    pixels[di] += static_cast<int64_t>(sites.size());
    const Grid<int64_t> ref = brute_force(sites.view());
    const Grid<int64_t> ref_d2 = distance_sq(ref.view());
    const Grid<int64_t> init = init_nearest(sites.view());
    const bool any_site = ref.size() > 0 && ref.data()[0] != kNoSite;

    for (std::size_t vi = 0; vi < kVariants.size(); ++vi) {
      const std::vector<int> steps = step_schedule(sites.height(), sites.width(), kVariants[vi]);
      const Grid<int64_t> out = run_steps(init.view(), steps);
      const Grid<int64_t> d2 = distance_sq(out.view());
      for (std::size_t p = 0; p < out.size(); ++p) {
        const int64_t s = out.data()[p];
        // Assigned iff there is a site; only to a real site; sites keep
        // themselves; one-sided error (never closer than the true nearest).
        const bool ok = any_site ? s >= 0 && sites.data()[static_cast<std::size_t>(s)] != 0 &&
                                       (sites.data()[p] == 0 || s == static_cast<int64_t>(p)) &&
                                       d2.data()[p] >= ref_d2.data()[p]
                                 : s == kNoSite;
        invariant_failures[vi] += ok ? 0 : 1;
        if (d2.data()[p] != ref_d2.data()[p]) {
          ++wrong[vi][di];
          if (kVariants[vi] == Variant::JfaPlus2 && first_bad_jfa2 < 0) {
            first_bad_jfa2 = i;
          }
        } else if (kVariants[vi] == Variant::JfaPlus2 && s != ref.data()[p]) {
          ++label_differs_jfa2;
        }
      }
    }
  }

  int64_t total_pixels = 0;
  for (const int64_t n : pixels) {
    total_pixels += n;
  }
  std::array<int64_t, kVariants.size()> total_wrong{};
  for (std::size_t vi = 0; vi < kVariants.size(); ++vi) {
    std::cout << "[ residual ] " << kVariantNames[vi] << ":";
    for (std::size_t di = 0; di < kDensities.size(); ++di) {
      total_wrong[vi] += wrong[vi][di];
      std::cout << "  d=" << kDensities[di] << " "
                << static_cast<double>(wrong[vi][di]) / static_cast<double>(pixels[di]);
    }
    const double rate = static_cast<double>(total_wrong[vi]) / static_cast<double>(total_pixels);
    std::cout << "  | all " << total_wrong[vi] << "/" << total_pixels << " = " << rate << "\n";
    RecordProperty(std::string("error_rate_") + kVariantNames[vi], std::to_string(rate));
    EXPECT_EQ(invariant_failures[vi], 0) << kVariantNames[vi];
  }
  std::cout << "[ residual ] jfa+2 pixels with a tied, different site: " << label_differs_jfa2
            << "\n";

  // jfa+2 is not exact (see JfaPlus2MinimalCounterexample): the Python
  // reference got about 9e-6 of pixels wrong over 18,334 random grids of this
  // kind, mostly at low density. These 1,000 happen to have no error, but that
  // is not a guarantee, so only the invariants above are exact; the error
  // rate is recorded and bounded.
  EXPECT_LT(static_cast<double>(total_wrong[2]) / static_cast<double>(total_pixels), 1e-4)
      << "first grid with a jfa+2 error: " << first_bad_jfa2;
  // Plain JFA's residual error is recorded, not pinned; only a sane bound
  // (Q1 measured at most 1.6e-4 per configuration).
  EXPECT_LT(static_cast<double>(total_wrong[0]) / static_cast<double>(total_pixels), 0.01);
  // As in test_error_rate_bounded_and_repaired_by_extra_pass.
  EXPECT_LE(total_wrong[1], total_wrong[0]);
}

TEST(OracleAgreement, JfaPlus2MinimalCounterexample) {
  // jfa+2 is approximate. An exhaustive search over every two-site grid found
  // no jfa+2 distance error below 156 px; this is the first one at 156:
  // 12 x 13, sites 39 = (3, 0) and 151 = (11, 8). Pixel 12 = (0, 12) is at
  // d2 = 137 from site 151 but keeps site 39 (d2 = 153).
  // The expected wrong pixels per variant are the Python reference's
  // (jfa_numpy.run_steps), so this also anchors C++/Python parity.
  Grid<uint8_t> sites(12, 13, 0);
  sites.data()[39] = 1;
  sites.data()[151] = 1;
  const Grid<int64_t> ref = brute_force(sites.view());
  const Grid<int64_t> ref_d2 = distance_sq(ref.view());
  const Grid<int64_t> init = init_nearest(sites.view());

  struct Case {
    Variant variant;
    std::vector<std::size_t> wrong;  // flat indices, all assigned site 39 instead of 151
  };
  const std::array<Case, 4> cases{{
      {Variant::Jfa, {12, 24, 25, 36, 37, 38, 48, 49, 50, 51}},
      {Variant::JfaPlus1, {12, 24, 25, 36, 37, 38}},
      {Variant::JfaPlus2, {12}},
      {Variant::OnePlusJfa, {}},
  }};
  for (const Case& c : cases) {
    const Grid<int64_t> out = run_steps(init.view(), step_schedule(12, 13, c.variant));
    const Grid<int64_t> d2 = distance_sq(out.view());
    std::vector<std::size_t> wrong;
    for (std::size_t p = 0; p < out.size(); ++p) {
      if (d2.data()[p] != ref_d2.data()[p]) {
        wrong.push_back(p);
        EXPECT_EQ(out.data()[p], 39) << p;
        EXPECT_EQ(ref.data()[p], 151) << p;
      }
    }
    EXPECT_EQ(wrong, c.wrong) << static_cast<int>(c.variant);
  }
  EXPECT_EQ(ref_d2.data()[12], 137);
}

}  // namespace
}  // namespace voronav
