// Roadmap tests: the heap's tie order, source_field, gvd_mask's rule,
// connect_pieces, degenerate maps, determinism across threads, and exact
// outputs pinned from roadmap.py (two_rooms, corridor) so this file checks
// Python parity without Python. tests/test_cpp_roadmap_parity.py is the rest.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/grid_search.hpp"
#include "voronav/jfa.hpp"
#include "voronav/roadmap.hpp"
#include "voronav/thread_pool.hpp"

#include "test_random.hpp"

using voronav::BinaryHeap;
using voronav::Grid;
using voronav::GridView;
using voronav::RepairResult;
using voronav::Roadmap;
using voronav::RoadmapParams;
using voronav::Schedule;
using voronav::Variant;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

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

template <typename T>
Grid<T> grid_of(const std::vector<std::vector<T>>& rows) {
  Grid<T> g(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()));
  for (int y = 0; y < g.height(); ++y) {
    for (int x = 0; x < g.width(); ++x) {
      g(y, x) = rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
    }
  }
  return g;
}

// FNV-1a over the little-endian bytes of the cells, for pinning whole grids.
template <typename T>
uint64_t fnv(const Grid<T>& g) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < g.size(); ++i) {
    const auto cell = static_cast<uint64_t>(g.data()[i]);
    for (std::size_t b = 0; b < sizeof(T); ++b) {
      h = (h ^ ((cell >> (8U * b)) & 0xFFU)) * 0x100000001b3ULL;
    }
  }
  return h;
}

// maps.corridor: free rows [(h - gap) // 2, + gap), obstacle elsewhere.
Grid<uint8_t> corridor(int h, int w, int gap) {
  Grid<uint8_t> g(h, w, 1);
  const int top = (h - gap) / 2;
  for (int y = top; y < top + gap; ++y) {
    for (int x = 0; x < w; ++x) {
      g(y, x) = 0;
    }
  }
  return g;
}

// map_sources.two_rooms(60, 100): the Python's rounding worked out by hand.
Grid<uint8_t> two_rooms_60_100() {
  Grid<uint8_t> g(60, 100, 1);
  const auto fill = [&g](int y0, int y1, int x0, int x1, uint8_t v) {
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        g(y, x) = v;
      }
    }
  };
  fill(3, 57, 3, 48, 0);
  fill(3, 57, 52, 97, 0);
  fill(27, 33, 48, 52, 0);  // door
  fill(10, 22, 15, 30, 1);
  fill(35, 50, 70, 75, 1);
  return g;
}

Grid<uint8_t> random_rooms(std::mt19937_64& rng, int h, int w, int boxes, double border_p) {
  Grid<uint8_t> g(h, w);
  for (int i = 0; i < boxes; ++i) {
    const int y = voronav::test::uniform_int(rng, 0, h - 1);
    const int x = voronav::test::uniform_int(rng, 0, w - 1);
    const int dy = voronav::test::uniform_int(rng, 1, std::max(1, h / 3));
    const int dx = voronav::test::uniform_int(rng, 1, std::max(1, w / 3));
    for (int yy = y; yy < std::min(h, y + dy); ++yy) {
      for (int xx = x; xx < std::min(w, x + dx); ++xx) {
        g(yy, xx) = 1;
      }
    }
  }
  if (voronav::test::bernoulli(rng, border_p)) {
    for (int x = 0; x < w; ++x) {
      g(0, x) = g(h - 1, x) = 1;
    }
    for (int y = 0; y < h; ++y) {
      g(y, 0) = g(y, w - 1) = 1;
    }
  }
  return g;
}

RoadmapParams params_for(double robot_radius, double bridge_lam = 0.0,
                         Variant variant = Variant::JfaPlus2) {
  RoadmapParams params;
  params.robot_radius = robot_radius;
  params.bridge_lam = bridge_lam;
  params.variant = variant;
  return params;
}

void expect_same_roadmap(const Roadmap& a, const Roadmap& b, const std::string& msg) {
  EXPECT_EQ(a.nearest, b.nearest) << msg;
  EXPECT_EQ(a.clearance, b.clearance) << msg;
  EXPECT_EQ(a.safe, b.safe) << msg;
  EXPECT_EQ(a.skeleton, b.skeleton) << msg;
  EXPECT_EQ(a.bridges, b.bridges) << msg;
  EXPECT_EQ(a.n_pruned, b.n_pruned) << msg;
  EXPECT_EQ(a.n_bridges, b.n_bridges) << msg;
  EXPECT_EQ(a.bridge_px, b.bridge_px) << msg;
}

}  // namespace

// The expected order comes from grid_search._push / _pop on the same inputs.
TEST(BinaryHeap, EqualKeysComeOutInTheNumbaOrder) {
  BinaryHeap heap;
  const std::vector<std::pair<double, int64_t>> pushes{{2.0, 10}, {1.0, 11}, {2.0, 12}, {1.0, 13},
                                                       {3.0, 14}, {1.0, 15}, {2.0, 16}, {1.0, 17},
                                                       {0.5, 18}, {2.0, 19}, {1.0, 20}};
  for (const auto& [key, value] : pushes) {
    heap.push(key, value);
  }
  const std::vector<std::pair<double, int64_t>> expected{{0.5, 18}, {1.0, 11}, {1.0, 13}, {1.0, 17},
                                                         {1.0, 20}, {1.0, 15}, {2.0, 12}, {2.0, 19},
                                                         {2.0, 10}, {2.0, 16}, {3.0, 14}};
  for (const auto& [key, value] : expected) {
    ASSERT_FALSE(heap.empty());
    const auto top = heap.pop();
    EXPECT_EQ(top.key, key);
    EXPECT_EQ(top.value, value);
  }
  EXPECT_TRUE(heap.empty());
}

TEST(SourceField, EqualCostsGoToTheEarlierSource) {
  const Grid<uint8_t> passable(1, 3, 1);
  const auto labels = grid_of<int64_t>({{1, 0, 2}});
  const auto field = voronav::source_field(passable.view(), labels.view());
  EXPECT_EQ(field.cost, grid_of<double>({{0.0, 1.0, 0.0}}));
  EXPECT_EQ(field.parent, (std::vector<int64_t>{-1, 0, -1}));
  EXPECT_EQ(field.owner, grid_of<int64_t>({{1, 1, 2}}));
  EXPECT_EQ(field.contacts.u, (std::vector<int64_t>{1}));
  EXPECT_EQ(field.contacts.v, (std::vector<int64_t>{2}));
  EXPECT_EQ(field.contacts.cost, (std::vector<double>{2.0}));
}

TEST(SourceField, DiagonalsCannotCutCorners) {
  const auto passable = from_rows({"#.", ".#"});
  const auto labels = grid_of<int64_t>({{1, 0}, {0, 0}});
  const auto field = voronav::source_field(passable.view(), labels.view());
  EXPECT_EQ(field.owner, grid_of<int64_t>({{1, 0}, {0, 0}}));
  EXPECT_EQ(field.cost, grid_of<double>({{0.0, kInf}, {kInf, kInf}}));
  EXPECT_TRUE(field.contacts.u.empty());
}

TEST(SourceField, DiagonalMovesCostSqrt2) {
  const auto passable = from_rows({"##", "##"});
  const auto labels = grid_of<int64_t>({{1, 0}, {0, 0}});
  const auto field = voronav::source_field(passable.view(), labels.view());
  EXPECT_EQ(field.cost(1, 1), std::sqrt(2.0));
  EXPECT_EQ(field.parent[3], 0);
}

TEST(SourceField, MultipliersCostTheEnteredPixelAndAverageAtContacts) {
  const Grid<uint8_t> passable(1, 4, 1);
  const auto labels = grid_of<int64_t>({{1, 0, 0, 2}});
  const auto mult = grid_of<double>({{1.0, 3.0, 5.0, 1.0}});
  const auto field =
      voronav::source_field(passable.view(), labels.view(), std::as_const(mult).view());
  // From the left 3 enters (0,1); from the right 5 enters (0,2).
  EXPECT_EQ(field.cost, grid_of<double>({{0.0, 3.0, 5.0, 0.0}}));
  EXPECT_EQ(field.owner, grid_of<int64_t>({{1, 1, 2, 2}}));
  ASSERT_EQ(field.contacts.u.size(), 1U);
  EXPECT_EQ(field.contacts.u[0], 1);
  EXPECT_EQ(field.contacts.v[0], 2);
  EXPECT_EQ(field.contacts.cost[0], 3.0 + 1.0 * 0.5 * (3.0 + 5.0) + 5.0);
}

TEST(SourceField, ContactsListRightDownThenDiagonals) {
  const auto passable = from_rows({"##", "##"});
  const auto labels = grid_of<int64_t>({{1, 2}, {3, 4}});
  const auto field = voronav::source_field(passable.view(), labels.view());
  EXPECT_EQ(field.contacts.u, (std::vector<int64_t>{0, 0, 0, 1, 1, 2}));
  EXPECT_EQ(field.contacts.v, (std::vector<int64_t>{1, 2, 3, 3, 2, 3}));
}

TEST(SourceField, ValidatesMultipliersOnlyOnPassablePixels) {
  const auto passable = from_rows({"#."});
  const auto labels = grid_of<int64_t>({{1, 0}});
  auto mult = grid_of<double>({{1.0, 0.5}});
  EXPECT_NO_THROW(
      (void)voronav::source_field(passable.view(), labels.view(), std::as_const(mult).view()));
  mult(0, 0) = 0.5;
  EXPECT_THROW(
      (void)voronav::source_field(passable.view(), labels.view(), std::as_const(mult).view()),
      std::invalid_argument);
  mult(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      (void)voronav::source_field(passable.view(), labels.view(), std::as_const(mult).view()),
      std::invalid_argument);
  const Grid<int64_t> wrong(2, 2);
  EXPECT_THROW((void)voronav::source_field(passable.view(), wrong.view()), std::invalid_argument);
}

TEST(BoundarySites, KeepsOnlyObstaclePixelsNextToFreeSpace) {
  const auto occ = from_rows({".....", ".###.", ".###.", ".###.", "....."});
  EXPECT_EQ(voronav::boundary_sites(occ.view()),
            from_rows({".....", ".###.", ".#.#.", ".###.", "....."}));
  const auto row = from_rows({"##.#"});
  EXPECT_EQ(voronav::boundary_sites(row.view()), from_rows({".#.#"}));
  const Grid<uint8_t> all(4, 4, 1);
  EXPECT_EQ(voronav::boundary_sites(all.view()), Grid<uint8_t>(4, 4));
  const auto diagonal = from_rows({"#.", ".#"});
  EXPECT_EQ(voronav::boundary_sites(diagonal.view()), diagonal);
}

TEST(ObstacleIds, FrameTakesTheLargestTouchingIdOrOnePerSide) {
  const auto occ = from_rows({"#######", "#.#.#.#", "#.....#", "#.....#", "#######"});
  EXPECT_EQ(voronav::obstacle_ids(occ.view()), grid_of<int64_t>({{3, 1, 1, 2, 2, 2, 3},
                                                                 {5, 0, 1, 0, 2, 0, 6},
                                                                 {5, 0, 0, 0, 0, 0, 6},
                                                                 {5, 0, 0, 0, 0, 0, 6},
                                                                 {4, 4, 4, 4, 4, 4, 4}}));
}

TEST(ObstacleIds, SmallGridsAreLabeledDirectly) {
  EXPECT_EQ(voronav::obstacle_ids(from_rows({"#.#"}).view()), grid_of<int64_t>({{1, 0, 2}}));
  EXPECT_EQ(voronav::obstacle_ids(from_rows({"##", "##"}).view()),
            grid_of<int64_t>({{1, 1}, {1, 1}}));
  EXPECT_EQ(voronav::obstacle_ids(from_rows({"#.", ".#"}).view()),
            grid_of<int64_t>({{1, 0}, {0, 1}}));
}

// gvd_mask's rule on hand-built fields. nearest and d2 need not be a real JFA
// output, which isolates the rule.
namespace {
Grid<uint8_t> gvd(const char* occ_row, std::vector<int64_t> nearest, std::vector<int64_t> d2,
                  double angle, double slack) {
  const auto occ = from_rows({occ_row});
  const auto w = static_cast<int>(nearest.size());
  Grid<int64_t> n(1, w);
  Grid<int64_t> d(1, w);
  for (int x = 0; x < w; ++x) {
    n(0, x) = nearest[static_cast<std::size_t>(x)];
    d(0, x) = d2[static_cast<std::size_t>(x)];
  }
  return voronav::gvd_mask(occ.view(), n.view(), d.view(), angle, slack);
}
}  // namespace

TEST(GvdMask, DifferentObstaclesAlwaysMakeAnEdge) {
  EXPECT_EQ(gvd("#..#", {0, 0, 3, 3}, {0, 1, 1, 0}, 60.0, 1000.0), from_rows({".##."}));
}

TEST(GvdMask, AdjacentAndDiagonalWallPixelsNeverMakeAnEdge) {
  // limit 0 with these parameters, so only the separation filter rejects it
  EXPECT_EQ(gvd("##....", {0, 0, 0, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, 180.0, -2.0),
            Grid<uint8_t>(1, 6));
  // separation^2 = 4 passes the filter; same obstacle, so the angle test decides
  EXPECT_EQ(gvd("###...", {0, 0, 0, 0, 2, 2}, {0, 0, 0, 1, 1, 1}, 180.0, -0.5),
            from_rows({"...##."}));
}

TEST(GvdMask, WideAngleTestIsStrictAndUsesTheLargerClearance) {
  // sep^2 = 4, limit = 2 * 1 + 0 = 2: 4 > 4 is false
  EXPECT_EQ(gvd("###...", {0, 0, 0, 0, 2, 2}, {0, 0, 0, 1, 1, 1}, 180.0, 0.0), Grid<uint8_t>(1, 6));
  // d is the larger of the pair's clearances (d2 = 4 at x = 4): limit 4
  EXPECT_EQ(gvd("###...", {0, 0, 0, 0, 2, 2}, {0, 0, 0, 1, 4, 1}, 180.0, 0.0), Grid<uint8_t>(1, 6));
  // The floating-point pin: 2 sin(30 degrees) evaluates to 0.9999999999999999,
  // so the limit is a hair under 2 and 4 > limit^2 holds. roadmap.py marks it too.
  EXPECT_EQ(gvd("###...", {0, 0, 0, 0, 2, 2}, {0, 0, 0, 4, 4, 1}, 60.0, 0.0),
            from_rows({"...##."}));
}

TEST(GvdMask, OnlyFreePixelsWithSitesMakeEdges) {
  // the pair crosses a pixel with no site, then an occupied pixel
  EXPECT_EQ(gvd("#..#", {0, -1, 3, 3}, {0, -1, 1, 0}, 60.0, 0.0), Grid<uint8_t>(1, 4));
}

TEST(GvdMask, RejectsBadArguments) {
  const Grid<uint8_t> occ(2, 2);
  const Grid<int64_t> n(2, 2, voronav::kNoSite);
  const Grid<int64_t> d(2, 2, -1);
  for (const double bad : {0.0, -1.0, 180.5, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW((void)voronav::gvd_mask(occ.view(), n.view(), d.view(), bad, 0.0),
                 std::invalid_argument);
  }
  EXPECT_NO_THROW((void)voronav::gvd_mask(occ.view(), n.view(), d.view(), 180.0, 0.0));
  const Grid<int64_t> wrong(2, 3);
  EXPECT_THROW((void)voronav::gvd_mask(occ.view(), wrong.view(), d.view()), std::invalid_argument);
}

// Two straight pieces five apart: the shortest path between them is the only
// one, so the expected bridge is exact.
TEST(ConnectPieces, BridgesPiecesInOneRegion) {
  const auto skeleton =
      from_rows({"............", "............", ".####..####.", "............", "............"});
  const Grid<uint8_t> safe(5, 12, 1);
  const RepairResult out = voronav::connect_pieces(skeleton.view(), safe.view(), 1);
  EXPECT_EQ(out.n_bridges, 1);
  EXPECT_EQ(out.bridge_px, 2);
  EXPECT_EQ(out.n_pruned, 0);
  EXPECT_EQ(out.skeleton, from_rows({"............", "............", ".##########.", "............",
                                     "............"}));
  EXPECT_EQ(out.bridges, from_rows({"............", "............", ".....##.....", "............",
                                    "............"}));
}

TEST(ConnectPieces, KeepsTheLargestPieceAndDropsTinyOnes) {
  const auto skeleton = from_rows({"###..#####"});
  const Grid<uint8_t> safe(1, 10, 1);
  const RepairResult out = voronav::connect_pieces(skeleton.view(), safe.view(), 8);
  EXPECT_EQ(out.n_pruned, 3);
  EXPECT_EQ(out.n_bridges, 0);
  EXPECT_EQ(out.skeleton, from_rows({".....#####"}));
}

TEST(ConnectPieces, SizeTiesKeepTheLowestLabel) {
  const auto skeleton = from_rows({"##..##"});
  const Grid<uint8_t> safe(1, 6, 1);
  const RepairResult out = voronav::connect_pieces(skeleton.view(), safe.view(), 8);
  EXPECT_EQ(out.n_pruned, 2);
  EXPECT_EQ(out.skeleton, from_rows({"##...."}));
}

TEST(ConnectPieces, DoesNotBridgeAcrossRegions) {
  const auto skeleton = from_rows({"##.#.##"});
  const auto safe = from_rows({"###.###"});
  const RepairResult out = voronav::connect_pieces(skeleton.view(), safe.view(), 1);
  EXPECT_EQ(out.n_bridges, 0);
  EXPECT_EQ(out.skeleton, skeleton);
}

TEST(ConnectPieces, ValidatesLambda) {
  const Grid<uint8_t> mask(2, 2, 1);
  EXPECT_THROW((void)voronav::connect_pieces(mask.view(), mask.view(), 8, std::nullopt, -1.0),
               std::invalid_argument);
  EXPECT_THROW((void)voronav::connect_pieces(mask.view(), mask.view(), 8, std::nullopt, 1.0),
               std::invalid_argument);
}

// A clearance-weighted bridge should stay away from a wall. Two pieces in a
// room; with bridge_lam > 0 the path takes the higher-clearance row even
// though a straight one exists through the low-clearance row.
TEST(ConnectPieces, WeightedBridgeAvoidsLowClearance) {
  const auto skeleton =
      from_rows({"...........", ".###...###.", "...........", "...........", "..........."});
  const Grid<uint8_t> safe(5, 11, 1);
  Grid<double> clearance(5, 11, 100.0);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 11; ++x) {
      clearance(y, x) = 0.1;
    }
  }
  clearance(1, 3) = clearance(1, 7) = 100.0;  // the piece ends are clear
  const auto plain = voronav::connect_pieces(skeleton.view(), safe.view(), 1);
  EXPECT_EQ(plain.skeleton(1, 5), 1);
  const auto weighted = voronav::connect_pieces(skeleton.view(), safe.view(), 1,
                                                std::as_const(clearance).view(), 64.0);
  EXPECT_EQ(weighted.n_bridges, 1);
  EXPECT_EQ(weighted.skeleton(1, 5), 0);
  EXPECT_EQ(weighted.skeleton(2, 5), 1);
}

TEST(BuildRoadmap, EmptyMapHasNoSites) {
  const Grid<uint8_t> occ(7, 9);
  const Roadmap r = voronav::build_roadmap(occ.view(), params_for(2.0));
  EXPECT_EQ(r.nearest, Grid<int64_t>(7, 9, voronav::kNoSite));
  EXPECT_EQ(r.clearance, Grid<double>(7, 9, kInf));
  EXPECT_EQ(r.safe, Grid<uint8_t>(7, 9, 1));
  EXPECT_EQ(r.skeleton, Grid<uint8_t>(7, 9));
  EXPECT_EQ(r.n_bridges, 0);
  EXPECT_EQ(r.robot_radius, 2.0);
  EXPECT_GE(r.build_s, 0.0);
}

TEST(BuildRoadmap, AllOccupiedMapHasNoSitesEither) {
  const Grid<uint8_t> occ(4, 5, 1);
  const Roadmap r = voronav::build_roadmap(occ.view());
  EXPECT_EQ(r.nearest, Grid<int64_t>(4, 5, voronav::kNoSite));
  EXPECT_EQ(r.safe, Grid<uint8_t>(4, 5));
  EXPECT_EQ(r.skeleton, Grid<uint8_t>(4, 5));
}

TEST(BuildRoadmap, OneObstacleHasNoSkeleton) {
  Grid<uint8_t> occ(9, 11);
  occ(4, 5) = 1;
  const Roadmap r = voronav::build_roadmap(occ.view());
  EXPECT_EQ(r.nearest, Grid<int64_t>(9, 11, 4 * 11 + 5));
  EXPECT_EQ(r.clearance(4, 5), 0.0);
  EXPECT_EQ(r.clearance(0, 5), 4.0);
  EXPECT_EQ(r.clearance(4, 8), 3.0);
  EXPECT_EQ(r.clearance(1, 1), 5.0);
  EXPECT_EQ(r.safe(4, 5), 0);
  EXPECT_EQ(r.safe(4, 6), 1);
  EXPECT_EQ(r.skeleton, Grid<uint8_t>(9, 11));
}

TEST(BuildRoadmap, OneRowMaps) {
  const auto occ = from_rows({"#......#"});
  const Roadmap r = voronav::build_roadmap(occ.view(), params_for(1.0));
  EXPECT_EQ(r.nearest, grid_of<int64_t>({{0, 0, 0, 0, 7, 7, 7, 7}}));
  EXPECT_EQ(r.clearance, grid_of<double>({{0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0}}));
  EXPECT_EQ(r.skeleton, from_rows({"...##..."}));
  const Grid<uint8_t> single(1, 1);
  EXPECT_EQ(voronav::build_roadmap(single.view()).skeleton, Grid<uint8_t>(1, 1));
  const Grid<uint8_t> single_occupied(1, 1, 1);
  EXPECT_EQ(voronav::build_roadmap(single_occupied.view()).safe, Grid<uint8_t>(1, 1));
  Grid<uint8_t> column(30, 1);
  column(11, 0) = 1;
  EXPECT_EQ(voronav::build_roadmap(column.view()).clearance(0, 0), 11.0);
}

// Beyond 46,340 px a squared distance overflows int32. d2 must stay exact.
TEST(BuildRoadmap, LargeDistancesDoNotOverflow) {
  constexpr int kWidth = 50000;
  Grid<uint8_t> occ(1, kWidth);
  occ(0, 0) = 1;
  const Roadmap r = voronav::build_roadmap(occ.view());
  EXPECT_EQ(r.clearance(0, kWidth - 1), static_cast<double>(kWidth - 1));
  EXPECT_EQ(r.clearance(0, 46341), 46341.0);
}

TEST(BuildRoadmap, RejectsBadInput) {
  const Grid<uint8_t> empty(0, 5);
  EXPECT_THROW((void)voronav::build_roadmap(empty.view()), std::invalid_argument);
  const Grid<uint8_t> empty_w(5, 0);
  EXPECT_THROW((void)voronav::build_roadmap(empty_w.view()), std::invalid_argument);
  const Grid<uint8_t> occ(3, 3);
  RoadmapParams bad_angle;
  bad_angle.min_angle_deg = 0.0;
  EXPECT_THROW((void)voronav::build_roadmap(occ.view(), bad_angle), std::invalid_argument);
  RoadmapParams bad_lam;
  bad_lam.bridge_lam = -1.0;
  EXPECT_THROW((void)voronav::build_roadmap(occ.view(), bad_lam), std::invalid_argument);
}

// Pinned from roadmap.py (robot_radius 2, bridge_lam 64, jfa+2): cross-language
// anchors, so this suite catches a parity break on its own.
TEST(BuildRoadmap, TwoRoomsMatchesPython) {
  const auto occ = two_rooms_60_100();
  const Roadmap r = voronav::build_roadmap(occ.view(), params_for(2.0, 64.0, Variant::JfaPlus2));
  EXPECT_EQ(r.n_bridges, 2);
  EXPECT_EQ(r.n_pruned, 0);
  EXPECT_EQ(r.bridge_px, 9);
  int64_t nodes = 0;
  for (const uint8_t cell : r.skeleton.view().flat()) {
    nodes += cell;
  }
  EXPECT_EQ(nodes, 559);
  EXPECT_EQ(fnv(r.skeleton), 0xd84830ce7f19e71aULL);
  EXPECT_EQ(fnv(r.nearest), 0x2de05272118258d1ULL);
}

TEST(BuildRoadmap, CorridorMatchesPython) {
  const auto occ = corridor(40, 80, 10);
  const Roadmap r = voronav::build_roadmap(occ.view(), params_for(1.0));
  int64_t nodes = 0;
  for (const uint8_t cell : r.skeleton.view().flat()) {
    nodes += cell;
  }
  EXPECT_EQ(nodes, 160);
  EXPECT_EQ(r.n_bridges, 0);
  EXPECT_EQ(r.n_pruned, 0);
  EXPECT_EQ(fnv(r.skeleton), 0x75f4be79b2cb01c5ULL);
  EXPECT_EQ(fnv(r.nearest), 0x874f4ffa827e04a5ULL);
  // both centerline rows of the corridor
  for (int x = 0; x < 80; ++x) {
    EXPECT_EQ(r.skeleton(19, x), 1) << x;
    EXPECT_EQ(r.skeleton(20, x), 1) << x;
  }
}

TEST(BuildRoadmap, SameResultForEveryThreadCountScheduleAndChunk) {
  std::mt19937_64 rng(21);
  for (int m = 0; m < 3; ++m) {
    const int h = voronav::test::uniform_int(rng, 32, 80);
    const int w = voronav::test::uniform_int(rng, 32, 80);
    const auto occ = random_rooms(rng, h, w, 12, 0.7);
    for (const Variant variant : {Variant::JfaPlus1, Variant::JfaPlus2}) {
      const RoadmapParams base = params_for(2.0, 64.0, variant);
      const Roadmap serial = voronav::build_roadmap(occ.view(), base);
      for (const unsigned threads : {2U, 7U, 10U}) {
        voronav::ThreadPool pool(threads);
        for (const Schedule schedule : {Schedule::Static, Schedule::Dynamic}) {
          for (const int chunk : {1, 4, 16}) {
            RoadmapParams params = base;
            params.schedule = schedule;
            params.chunk = chunk;
            const Roadmap r = voronav::build_roadmap(occ.view(), params, &pool);
            expect_same_roadmap(
                serial, r,
                "threads=" + std::to_string(threads) + " chunk=" + std::to_string(chunk));
          }
        }
      }
    }
  }
}

TEST(BuildRoadmap, RandomRoomsKeepTheInvariants) {
  std::mt19937_64 rng(22);
  for (int m = 0; m < 200; ++m) {
    const int h = voronav::test::uniform_int(rng, 4, 70);
    const int w = voronav::test::uniform_int(rng, 4, 70);
    const auto occ = random_rooms(rng, h, w, voronav::test::uniform_int(rng, 0, 14), 0.7);
    RoadmapParams params = params_for(1.5);
    params.min_piece_px = 3;
    const Roadmap r = voronav::build_roadmap(occ.view(), params);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        ASSERT_FALSE(r.skeleton(y, x) != 0 && r.safe(y, x) == 0);
        ASSERT_EQ(r.safe(y, x) != 0, occ(y, x) == 0 && r.clearance(y, x) >= 1.5);
      }
    }
  }
}
