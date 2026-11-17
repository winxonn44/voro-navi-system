#pragma once

// The generalized Voronoi diagram roadmap (checkpoint C5): a port of
// src/voronav/maps.py:boundary_sites and src/voronav/planning/roadmap.py, which
// is the spec. Every output of build_roadmap equals the Python one exactly
// (tests/test_cpp_roadmap_parity.py), so the two can be swapped.
//
// Doubles are compared exactly, so this file's source is compiled with
// -ffp-contract=off (see grid_search.hpp). Costs and thresholds are computed
// in the same order as the Python.

#include <cstdint>
#include <optional>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

namespace voronav {

// Obstacle pixels (nonzero) 4-adjacent to free space.
Grid<uint8_t> boundary_sites(GridView<const uint8_t> occ);

// Obstacle component ids (0 = free), 8-connected, ignoring the outer 1-px frame
// as a connector: frame pixels take the largest id around them (3x3, clamped at
// the map's edge) or, if none, one id per side of the map, top and bottom owning
// the corners. Grids under 3 px per side are labeled directly.
Grid<int64_t> obstacle_ids(GridView<const uint8_t> occ);

// Pixels on the generalized Voronoi diagram: a 4-adjacent pair of free pixels
// (p, q) with sites s_p, s_q is a GVD edge if |s_p - s_q|^2 > 3 and the sites
// lie on different obstacles or subtend a wide angle (|s_p - s_q| > 2 sin(theta
// / 2) * d + slack, d the larger clearance). Both pixels are marked. Throws
// std::invalid_argument on a shape mismatch or a min_angle_deg outside
// (0, 180].
Grid<uint8_t> gvd_mask(GridView<const uint8_t> occ, GridView<const int64_t> nearest,
                       GridView<const int64_t> d2, double min_angle_deg = 60.0,
                       double raster_slack_px = 2.0);

struct RepairResult {
  Grid<uint8_t> skeleton;
  Grid<uint8_t> bridges;  // pixels the bridges added
  int64_t n_pruned = 0;   // skeleton pixels dropped as tiny fragments
  int64_t n_bridges = 0;  // paths added to join pieces in one free region
  int64_t bridge_px = 0;  // pixels those bridges added
};

// One connected skeleton piece per free region: drops pieces under
// min_piece_px (never a region's largest), then joins the rest along a minimum
// spanning tree of cheapest contacts (Mehlhorn). bridge_lam > 0 weights steps by
// 1 + bridge_lam / clearance. Throws std::invalid_argument if bridge_lam < 0 or
// bridge_lam > 0 without a clearance field, and std::runtime_error if a kept
// piece cannot reach the rest of its region (impossible when skeleton is inside
// safe, as in build_roadmap).
RepairResult connect_pieces(GridView<const uint8_t> skeleton, GridView<const uint8_t> safe,
                            int64_t min_piece_px = 8,
                            std::optional<GridView<const double>> clearance = std::nullopt,
                            double bridge_lam = 0.0);

struct RoadmapParams {
  double robot_radius = 1.0;
  double min_angle_deg = 60.0;
  double raster_slack_px = 2.0;
  int64_t min_piece_px = 8;
  double bridge_lam = 0.0;
  Variant variant = Variant::JfaPlus2;
  Schedule schedule = Schedule::Dynamic;
  int chunk = kDefaultChunk;
};

struct Roadmap {
  Grid<int64_t> nearest;   // nearest boundary site (flat), kNoSite if none
  Grid<double> clearance;  // px; +inf if no site
  Grid<uint8_t> safe;      // free and clearance >= robot_radius
  Grid<uint8_t> skeleton;  // GVD pixels inside safe, repaired
  Grid<uint8_t> bridges;
  double robot_radius = 0.0;
  double build_s = 0.0;
  int64_t n_pruned = 0;
  int64_t n_bridges = 0;
  int64_t bridge_px = 0;
};

// JFA on the boundary sites (on `pool` if given; the result does not depend on
// threads, schedule or chunk), then extraction, serially. The JFA variant is a
// parameter because jfa+2 is not exact: parity holds per variant. Throws
// std::invalid_argument for an empty grid.
Roadmap build_roadmap(GridView<const uint8_t> occ, const RoadmapParams& params = {},
                      ThreadPool* pool = nullptr);

}  // namespace voronav
