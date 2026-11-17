#include "voronav/roadmap.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "voronav/grid_search.hpp"
#include "voronav/labeling.hpp"

namespace voronav {

namespace {

// Squared site separation must exceed this: adjacent and diagonal wall pixels
// (sep^2 = 1 or 2) never form a GVD edge, while a one-pixel gap between two
// obstacles (sites 2 px apart, sep^2 = 4) does.
constexpr int64_t kMinSep2 = 3;

template <typename A, typename B>
void require_same_shape(GridView<A> a, GridView<B> b, const char* who) {
  if (a.height != b.height || a.width != b.width) {
    throw std::invalid_argument(std::string(who) + ": arrays must have the same shape");
  }
}

}  // namespace

Grid<uint8_t> boundary_sites(GridView<const uint8_t> occ) {
  const int h = occ.height;
  const int w = occ.width;
  Grid<uint8_t> out(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (occ(y, x) == 0) {
        continue;
      }
      const bool near_free = (y > 0 && occ(y - 1, x) == 0) || (y + 1 < h && occ(y + 1, x) == 0) ||
                             (x > 0 && occ(y, x - 1) == 0) || (x + 1 < w && occ(y, x + 1) == 0);
      out(y, x) = near_free ? 1 : 0;
    }
  }
  return out;
}

Grid<int64_t> obstacle_ids(GridView<const uint8_t> occ) {
  const int h = occ.height;
  const int w = occ.width;
  if (h < 3 || w < 3) {
    return label_components(occ, Connectivity::Eight).labels;
  }
  Grid<uint8_t> inner(h, w);
  for (int y = 1; y + 1 < h; ++y) {
    for (int x = 1; x + 1 < w; ++x) {
      inner(y, x) = occ(y, x);
    }
  }
  const Labels inner_labels = label_components(inner.view(), Connectivity::Eight);
  const Grid<int64_t>& ids = inner_labels.labels;
  const int64_t n = inner_labels.count;
  Grid<int64_t> out = ids;
  const auto assign_frame = [&](int y, int x) {
    if (occ(y, x) == 0) {
      return;
    }
    int64_t touching = 0;  // maximum_filter(ids, size=3); reflect mode adds no new cells
    for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1); ++yy) {
      for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx) {
        touching = std::max(touching, ids(yy, xx));
      }
    }
    if (touching > 0) {
      out(y, x) = touching;
    } else if (y == 0) {
      out(y, x) = n + 1;
    } else if (y == h - 1) {
      out(y, x) = n + 2;
    } else if (x == 0) {
      out(y, x) = n + 3;
    } else {
      out(y, x) = n + 4;
    }
  };
  for (int x = 0; x < w; ++x) {
    assign_frame(0, x);
    assign_frame(h - 1, x);
  }
  for (int y = 1; y + 1 < h; ++y) {
    assign_frame(y, 0);
    assign_frame(y, w - 1);
  }
  return out;
}

Grid<uint8_t> gvd_mask(GridView<const uint8_t> occ, GridView<const int64_t> nearest,
                       GridView<const int64_t> d2, double min_angle_deg, double raster_slack_px) {
  if (!(min_angle_deg > 0.0 && min_angle_deg <= 180.0)) {
    throw std::invalid_argument("min_angle_deg must be in (0, 180]");
  }
  require_same_shape(occ, nearest, "gvd_mask");
  require_same_shape(occ, d2, "gvd_mask");
  const int h = occ.height;
  const int w = occ.width;
  const Grid<int64_t> obstacle = obstacle_ids(occ);
  const auto obstacle_flat = obstacle.view().flat();
  const auto site_obstacle = [&](int64_t site) {
    return obstacle_flat[static_cast<std::size_t>(site)];
  };
  const auto free_px = [&](int y, int x) { return occ(y, x) == 0 && nearest(y, x) >= 0; };
  // math.radians is x * (pi / 180), in that grouping
  const double radians = min_angle_deg * (std::numbers::pi / 180.0);
  const double rho = 2.0 * std::sin(radians / 2.0);
  const int64_t width = w;

  Grid<uint8_t> skel(h, w);
  const auto test_pair = [&](int py, int px, int qy, int qx) {
    if (!free_px(py, px) || !free_px(qy, qx)) {
      return;
    }
    const int64_t sp = nearest(py, px);
    const int64_t sq = nearest(qy, qx);
    const int64_t dy = sp / width - sq / width;
    const int64_t dx = sp % width - sq % width;
    const int64_t sep2 = dy * dy + dx * dx;
    if (sep2 <= kMinSep2) {
      return;
    }
    bool edge = site_obstacle(sp) != site_obstacle(sq);
    if (!edge) {
      const double dmax =
          std::sqrt(static_cast<double>(std::max({d2(py, px), d2(qy, qx), int64_t{0}})));
      const double limit = rho * dmax + raster_slack_px;
      edge = static_cast<double>(sep2) > limit * limit;
    }
    if (edge) {
      skel(py, px) = 1;
      skel(qy, qx) = 1;
    }
  };
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x + 1 < w; ++x) {
      test_pair(y, x, y, x + 1);
    }
  }
  for (int y = 0; y + 1 < h; ++y) {
    for (int x = 0; x < w; ++x) {
      test_pair(y, x, y + 1, x);
    }
  }
  return skel;
}

RepairResult connect_pieces(GridView<const uint8_t> skeleton, GridView<const uint8_t> safe,
                            int64_t min_piece_px, std::optional<GridView<const double>> clearance,
                            double bridge_lam) {
  if (bridge_lam < 0.0) {
    throw std::invalid_argument("bridge_lam must be >= 0");
  }
  if (bridge_lam > 0.0 && !clearance.has_value()) {
    throw std::invalid_argument("bridge_lam > 0 needs the clearance field");
  }
  require_same_shape(skeleton, safe, "connect_pieces");
  const int h = skeleton.height;
  const int w = skeleton.width;

  std::optional<Grid<double>> mult;
  if (bridge_lam > 0.0) {
    require_same_shape(skeleton, *clearance, "connect_pieces");
    mult.emplace(h, w, std::numeric_limits<double>::infinity());
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if ((*clearance)(y, x) > 0.0) {
          (*mult)(y, x) = 1.0 + bridge_lam / (*clearance)(y, x);
        }
      }
    }
  }

  RepairResult out{to_grid(skeleton), Grid<uint8_t>(h, w)};
  const Labels pieces = reachable_components(skeleton, safe);
  const int64_t n_pieces = pieces.count;
  if (n_pieces <= 1) {
    return out;
  }
  const Labels regions = reachable_components(safe);

  const auto n_slots = static_cast<std::size_t>(n_pieces) + 1;
  std::vector<int64_t> size(n_slots, 0);
  std::vector<int64_t> region_of(n_slots, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const auto p = static_cast<std::size_t>(pieces.labels(y, x));
      ++size[p];
      if (skeleton(y, x) != 0) {
        region_of[p] = regions.labels(y, x);
      }
    }
  }

  // The largest piece of each region (ties: lowest label) is always kept.
  std::vector<int64_t> order(static_cast<std::size_t>(n_pieces));
  for (int64_t i = 0; i < n_pieces; ++i) {
    order[static_cast<std::size_t>(i)] = i + 1;
  }
  std::ranges::stable_sort(order, [&](int64_t a, int64_t b) {
    const auto ia = static_cast<std::size_t>(a);
    const auto ib = static_cast<std::size_t>(b);
    return std::make_pair(region_of[ia], -size[ia]) < std::make_pair(region_of[ib], -size[ib]);
  });
  std::vector<uint8_t> keep(n_slots, 0);
  for (std::size_t p = 1; p < n_slots; ++p) {
    keep[p] = size[p] >= min_piece_px ? 1 : 0;
  }
  int64_t previous_region = -1;
  for (const int64_t p : order) {
    const int64_t r = region_of[static_cast<std::size_t>(p)];
    if (r != previous_region) {
      keep[static_cast<std::size_t>(p)] = 1;
      previous_region = r;
    }
  }

  std::vector<int64_t> kept_regions;
  int64_t n_kept = 0;
  for (std::size_t p = 1; p < n_slots; ++p) {
    if (keep[p] != 0) {
      ++n_kept;
      kept_regions.push_back(region_of[p]);
    }
  }
  std::ranges::sort(kept_regions);
  const auto last_unique = std::ranges::unique(kept_regions);
  const auto n_tree_regions = static_cast<int64_t>(last_unique.begin() - kept_regions.begin());

  Grid<int64_t> sources(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int64_t p = pieces.labels(y, x);
      if (keep[static_cast<std::size_t>(p)] != 0) {
        sources(y, x) = p;
      } else if (skeleton(y, x) != 0) {
        out.skeleton(y, x) = 0;
        ++out.n_pruned;
      }
    }
  }
  if (n_kept == n_tree_regions) {
    return out;  // one piece per region already
  }

  std::optional<GridView<const double>> mult_view;
  if (mult.has_value()) {
    mult_view = std::as_const(*mult).view();
  }
  const SourceField field = source_field(safe, sources.view(), mult_view);
  const Contacts& contacts = field.contacts;

  // Cheapest contact per pair of pieces (ties: first in scan order), sorted by
  // (lo, hi) as the CSR matrix SciPy's spanning tree reads.
  struct Contact {
    int64_t lo;
    int64_t hi;
    double cost;
    std::size_t index;
  };
  const auto owner_flat = field.owner.view().flat();
  std::vector<Contact> sorted_contacts;
  sorted_contacts.reserve(contacts.u.size());
  for (std::size_t k = 0; k < contacts.u.size(); ++k) {
    const int64_t a = owner_flat[static_cast<std::size_t>(contacts.u[k])];
    const int64_t b = owner_flat[static_cast<std::size_t>(contacts.v[k])];
    sorted_contacts.push_back({std::min(a, b), std::max(a, b), contacts.cost[k], k});
  }
  std::ranges::sort(sorted_contacts, [](const Contact& a, const Contact& b) {
    return std::tie(a.lo, a.hi, a.cost, a.index) < std::tie(b.lo, b.hi, b.cost, b.index);
  });
  std::vector<std::size_t> best;
  std::vector<WeightedEdge> edges;
  for (const Contact& c : sorted_contacts) {
    if (edges.empty() || edges.back().u != c.lo || edges.back().v != c.hi) {
      best.push_back(c.index);
      edges.push_back({c.lo, c.hi, c.cost});
    }
  }
  const std::vector<std::size_t> tree = minimum_spanning_tree(n_pieces + 1, edges);
  if (static_cast<int64_t>(tree.size()) != n_kept - n_tree_regions) {
    throw std::runtime_error("skeleton piece unreachable inside its own region");
  }

  // Bridges in (lo, hi) order: which bridge first covers a shared pixel decides
  // bridge_px. A chain's pixels are distinct: each end stays in its own territory.
  const auto flat_skeleton = out.skeleton.view().flat();
  const auto flat_bridges = out.bridges.view().flat();
  for (const std::size_t edge : tree) {
    const std::size_t k = best[edge];
    for (int64_t node : {contacts.u[k], contacts.v[k]}) {
      while (node >= 0) {
        const auto px = static_cast<std::size_t>(node);
        if (flat_skeleton[px] == 0) {
          flat_skeleton[px] = 1;
          flat_bridges[px] = 1;
          ++out.bridge_px;
        }
        node = field.parent[px];
      }
    }
    ++out.n_bridges;
  }
  return out;
}

Roadmap build_roadmap(GridView<const uint8_t> occ, const RoadmapParams& params, ThreadPool* pool) {
  if (occ.size() == 0) {
    throw std::invalid_argument("build_roadmap: empty grid");
  }
  const auto t0 = std::chrono::steady_clock::now();
  const int h = occ.height;
  const int w = occ.width;
  const Grid<uint8_t> sites = boundary_sites(occ);
  const Grid<int64_t> init = init_nearest(sites.view());
  const std::vector<int> steps = step_schedule(h, w, params.variant);
  Roadmap road;
  road.nearest = run_steps(init.view(), steps, pool, params.schedule, params.chunk);
  const Grid<int64_t> d2 = distance_sq(road.nearest.view());

  road.clearance = Grid<double>(h, w, std::numeric_limits<double>::infinity());
  road.safe = Grid<uint8_t>(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (d2(y, x) >= 0) {
        road.clearance(y, x) = std::sqrt(static_cast<double>(d2(y, x)));
      }
      road.safe(y, x) = (occ(y, x) == 0 && road.clearance(y, x) >= params.robot_radius) ? 1 : 0;
    }
  }

  Grid<uint8_t> skeleton =
      gvd_mask(occ, road.nearest.view(), d2.view(), params.min_angle_deg, params.raster_slack_px);
  auto skeleton_flat = skeleton.view().flat();
  const auto safe_flat = std::as_const(road.safe).view().flat();
  for (std::size_t i = 0; i < skeleton_flat.size(); ++i) {
    skeleton_flat[i] = static_cast<uint8_t>(skeleton_flat[i] & safe_flat[i]);
  }

  RepairResult repair = connect_pieces(skeleton.view(), road.safe.view(), params.min_piece_px,
                                       std::as_const(road.clearance).view(), params.bridge_lam);
  road.skeleton = std::move(repair.skeleton);
  road.bridges = std::move(repair.bridges);
  road.robot_radius = params.robot_radius;
  road.n_pruned = repair.n_pruned;
  road.n_bridges = repair.n_bridges;
  road.bridge_px = repair.bridge_px;
  road.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return road;
}

}  // namespace voronav
