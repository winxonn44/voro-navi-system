// Labeling and the spanning tree: hand-checked cases, plus random grids and
// graphs against independent reference implementations here in C++ (flood fill,
// quadratic Kruskal). tests/test_cpp_roadmap_parity.py compares them with SciPy.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/labeling.hpp"

#include "test_random.hpp"

using voronav::Connectivity;
using voronav::DisjointSet;
using voronav::Grid;
using voronav::GridView;
using voronav::WeightedEdge;

namespace {

Grid<uint8_t> from_rows(const std::vector<std::string>& rows) {
  Grid<uint8_t> g(static_cast<int>(rows.size()),
                  rows.empty() ? 0 : static_cast<int>(rows[0].size()));
  for (int y = 0; y < g.height(); ++y) {
    for (int x = 0; x < g.width(); ++x) {
      g(y, x) = rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] == '#' ? 1 : 0;
    }
  }
  return g;
}

Grid<int64_t> ints(const std::vector<std::vector<int64_t>>& rows) {
  Grid<int64_t> g(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()));
  for (int y = 0; y < g.height(); ++y) {
    for (int x = 0; x < g.width(); ++x) {
      g(y, x) = rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
    }
  }
  return g;
}

Grid<uint8_t> random_mask(std::mt19937_64& rng, int h, int w, double p) {
  Grid<uint8_t> g(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      g(y, x) = voronav::test::bernoulli(rng, p) ? 1 : 0;
    }
  }
  return g;
}

// Reference: breadth-first flood fill, numbered by the first pixel in scan
// order. No union-find, so a bug there cannot hide here.
voronav::Labels flood_fill(const Grid<uint8_t>& mask, Connectivity connectivity) {
  const int h = mask.height();
  const int w = mask.width();
  voronav::Labels out{Grid<int64_t>(h, w), 0};
  const int radius = 1;
  const bool eight = connectivity == Connectivity::Eight;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (mask(y, x) == 0 || out.labels(y, x) != 0) {
        continue;
      }
      ++out.count;
      out.labels(y, x) = out.count;
      std::queue<std::pair<int, int>> open;
      open.emplace(y, x);
      while (!open.empty()) {
        const auto [uy, ux] = open.front();
        open.pop();
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dx = -radius; dx <= radius; ++dx) {
            if ((dy == 0 && dx == 0) || (!eight && dy != 0 && dx != 0)) {
              continue;
            }
            const int vy = uy + dy;
            const int vx = ux + dx;
            if (vy >= 0 && vy < h && vx >= 0 && vx < w && mask(vy, vx) != 0 &&
                out.labels(vy, vx) == 0) {
              out.labels(vy, vx) = out.count;
              open.emplace(vy, vx);
            }
          }
        }
      }
    }
  }
  return out;
}

const voronav::GridView<const uint8_t> view_of(const Grid<uint8_t>& g) { return g.view(); }

}  // namespace

TEST(DisjointSet, UnitesAndFinds) {
  DisjointSet sets(6);
  EXPECT_NE(sets.find(0), sets.find(1));
  EXPECT_TRUE(sets.unite(0, 1));
  EXPECT_FALSE(sets.unite(1, 0));
  EXPECT_TRUE(sets.unite(2, 3));
  EXPECT_TRUE(sets.unite(0, 3));
  EXPECT_EQ(sets.find(1), sets.find(2));
  EXPECT_NE(sets.find(0), sets.find(4));
}

TEST(LabelComponents, NumbersInScanOrderOfFirstPixel) {
  const auto occ = from_rows({"#..#", "...#", "#..."});
  for (const Connectivity c : {Connectivity::Four, Connectivity::Eight}) {
    const auto out = voronav::label_components(view_of(occ), c);
    EXPECT_EQ(out.count, 3);
    EXPECT_EQ(out.labels, ints({{1, 0, 0, 2}, {0, 0, 0, 2}, {3, 0, 0, 0}}));
  }
}

// The left arm joins the region only on the last row, after the second region
// has started: it must still take the first region's number.
TEST(LabelComponents, MergesKeepTheFirstPixelsNumber) {
  const auto occ = from_rows({"..#.#", "#.#.#", "###.#"});
  const auto out = voronav::label_components(view_of(occ), Connectivity::Eight);
  EXPECT_EQ(out.count, 2);
  EXPECT_EQ(out.labels, ints({{0, 0, 1, 0, 2}, {1, 0, 1, 0, 2}, {1, 1, 1, 0, 2}}));
}

TEST(LabelComponents, DiagonalsConnectOnlyWithEight) {
  const auto occ = from_rows({"#..", ".#.", "..#"});
  EXPECT_EQ(voronav::label_components(view_of(occ), Connectivity::Eight).count, 1);
  const auto four = voronav::label_components(view_of(occ), Connectivity::Four);
  EXPECT_EQ(four.count, 3);
  EXPECT_EQ(four.labels, ints({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));
  const auto anti = from_rows({"..#", ".#.", "#.."});
  EXPECT_EQ(voronav::label_components(view_of(anti), Connectivity::Eight).count, 1);
  EXPECT_EQ(voronav::label_components(view_of(anti), Connectivity::Four).count, 3);
}

TEST(LabelComponents, DegenerateMasks) {
  const Grid<uint8_t> empty(0, 0);
  EXPECT_EQ(voronav::label_components(view_of(empty), Connectivity::Four).count, 0);
  const Grid<uint8_t> none(3, 4);
  const auto zero = voronav::label_components(view_of(none), Connectivity::Eight);
  EXPECT_EQ(zero.count, 0);
  EXPECT_EQ(zero.labels, Grid<int64_t>(3, 4));
  const Grid<uint8_t> full(3, 4, 1);
  EXPECT_EQ(voronav::label_components(view_of(full), Connectivity::Four).count, 1);
  const auto row = from_rows({"#.##.#"});
  EXPECT_EQ(voronav::label_components(view_of(row), Connectivity::Eight).labels,
            ints({{1, 0, 2, 2, 0, 3}}));
}

TEST(LabelComponents, MatchesFloodFillOnRandomGrids) {
  std::mt19937_64 rng(11);
  for (int i = 0; i < 400; ++i) {
    const int h = voronav::test::uniform_int(rng, 1, 40);
    const int w = voronav::test::uniform_int(rng, 1, 40);
    const double p = static_cast<double>(voronav::test::uniform_int(rng, 1, 9)) / 10.0;
    const auto mask = random_mask(rng, h, w, p);
    for (const Connectivity c : {Connectivity::Four, Connectivity::Eight}) {
      const auto out = voronav::label_components(view_of(mask), c);
      const auto ref = flood_fill(mask, c);
      EXPECT_EQ(out.count, ref.count) << h << "x" << w;
      EXPECT_EQ(out.labels, ref.labels) << h << "x" << w;
    }
  }
}

// A diagonal step cannot cut a corner: both corner cells must be in corner_ok.
TEST(ReachableComponents, DiagonalsNeedBothCorners) {
  const auto mask = from_rows({"#.", ".#"});
  const auto none = from_rows({"..", ".."});
  const auto one = from_rows({".#", ".."});
  const auto both = from_rows({".#", "#."});
  EXPECT_EQ(voronav::reachable_components(view_of(mask), view_of(none)).count, 2);
  EXPECT_EQ(voronav::reachable_components(view_of(mask), view_of(one)).count, 2);
  EXPECT_EQ(voronav::reachable_components(view_of(mask), view_of(both)).count, 1);
}

TEST(ReachableComponents, AntiDiagonalNeedsItsOwnCorners) {
  const auto mask = from_rows({".#", "#."});
  const auto both = from_rows({"#.", ".#"});
  const auto wrong = from_rows({".#", "#."});  // the other diagonal's corners
  EXPECT_EQ(voronav::reachable_components(view_of(mask), view_of(both)).count, 1);
  EXPECT_EQ(voronav::reachable_components(view_of(mask), view_of(wrong)).count, 2);
}

// With corner_ok == mask this is 4-connectivity, the case that crashed a
// planner when regions were labeled with an 8-connected structure.
TEST(ReachableComponents, DefaultIsFourConnectivity) {
  const auto mask = from_rows({"#.", ".#"});
  EXPECT_EQ(voronav::reachable_components(view_of(mask)).count, 2);
  std::mt19937_64 rng(12);
  for (int i = 0; i < 200; ++i) {
    const int h = voronav::test::uniform_int(rng, 1, 30);
    const int w = voronav::test::uniform_int(rng, 1, 30);
    const auto m = random_mask(rng, h, w, 0.55);
    EXPECT_EQ(voronav::reachable_components(view_of(m)).labels,
              flood_fill(m, Connectivity::Four).labels);
    const Grid<uint8_t> full(h, w, 1);
    EXPECT_EQ(voronav::reachable_components(view_of(m), view_of(full)).labels,
              flood_fill(m, Connectivity::Eight).labels);
  }
}

TEST(ReachableComponents, RejectsBadShapes) {
  const Grid<uint8_t> a(2, 2);
  const Grid<uint8_t> b(2, 3);
  EXPECT_THROW((void)voronav::reachable_components(view_of(a), view_of(b)), std::invalid_argument);
  const Grid<uint8_t> empty(0, 4);
  EXPECT_THROW((void)voronav::reachable_components(view_of(empty)), std::invalid_argument);
}

TEST(MinimumSpanningTree, EqualWeightsPreferEarlierEdges) {
  const std::vector<WeightedEdge> triangle{{0, 1, 1.0}, {0, 2, 1.0}, {1, 2, 1.0}};
  EXPECT_EQ(voronav::minimum_spanning_tree(3, triangle), (std::vector<std::size_t>{0, 1}));
  const std::vector<WeightedEdge> heavy{{0, 1, 2.0}, {0, 2, 1.0}, {1, 2, 1.0}};
  EXPECT_EQ(voronav::minimum_spanning_tree(3, heavy), (std::vector<std::size_t>{1, 2}));
}

TEST(MinimumSpanningTree, ForestsAndDegenerateGraphs) {
  const std::vector<WeightedEdge> forest{{0, 1, 3.0}, {2, 3, 1.0}};
  EXPECT_EQ(voronav::minimum_spanning_tree(5, forest), (std::vector<std::size_t>{0, 1}));
  EXPECT_TRUE(voronav::minimum_spanning_tree(0, {}).empty());
  EXPECT_TRUE(voronav::minimum_spanning_tree(1, {}).empty());
  const std::vector<WeightedEdge> loop{{1, 1, 1.0}, {1, 2, 2.0}};
  EXPECT_EQ(voronav::minimum_spanning_tree(3, loop), (std::vector<std::size_t>{1}));
  const std::vector<WeightedEdge> inf{{0, 1, std::numeric_limits<double>::infinity()}};
  EXPECT_EQ(voronav::minimum_spanning_tree(2, inf), (std::vector<std::size_t>{0}));
}

TEST(MinimumSpanningTree, RejectsBadEdges) {
  const auto bad = [](std::vector<WeightedEdge> e) {
    EXPECT_THROW((void)voronav::minimum_spanning_tree(3, e), std::invalid_argument);
  };
  bad({{0, 3, 1.0}});
  bad({{-1, 1, 1.0}});
  bad({{0, 1, 0.0}});
  bad({{0, 1, -1.0}});
  bad({{0, 1, std::numeric_limits<double>::quiet_NaN()}});
  bad({{1, 2, 1.0}, {0, 1, 1.0}});  // out of order
  bad({{0, 1, 1.0}, {0, 1, 2.0}});  // duplicate
}

// Reference: quadratic Kruskal, relabeling a component vector instead of
// using union-find, trying edges in (weight, u, v) order.
TEST(MinimumSpanningTree, MatchesQuadraticKruskalOnRandomGraphsWithTies) {
  std::mt19937_64 rng(13);
  for (int round = 0; round < 300; ++round) {
    const int n = voronav::test::uniform_int(rng, 2, 40);
    std::map<std::pair<int64_t, int64_t>, double> by_pair;
    const int m = voronav::test::uniform_int(rng, 0, 3 * n);
    for (int i = 0; i < m; ++i) {
      const int a = voronav::test::uniform_int(rng, 0, n - 1);
      const int b = voronav::test::uniform_int(rng, 0, n - 1);
      if (a != b) {
        by_pair[{std::min(a, b), std::max(a, b)}] = voronav::test::uniform_int(rng, 1, 3);
      }
    }
    std::vector<WeightedEdge> edges;
    edges.reserve(by_pair.size());
    for (const auto& [pair, w] : by_pair) {
      edges.push_back({pair.first, pair.second, w});
    }
    std::vector<std::size_t> order(edges.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
      return std::tie(edges[a].weight, a) < std::tie(edges[b].weight, b);
    });
    std::vector<int> component(static_cast<std::size_t>(n));
    std::iota(component.begin(), component.end(), 0);
    std::vector<std::size_t> expected;
    for (const std::size_t i : order) {
      const int a = component[static_cast<std::size_t>(edges[i].u)];
      const int b = component[static_cast<std::size_t>(edges[i].v)];
      if (a != b) {
        expected.push_back(i);
        std::ranges::replace(component, b, a);
      }
    }
    std::ranges::sort(expected);
    EXPECT_EQ(voronav::minimum_spanning_tree(n, edges), expected);
  }
}
