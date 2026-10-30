#include "voronav/oracle.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"

namespace voronav {

Grid<int64_t> brute_force(GridView<const uint8_t> sites) {
  const int64_t w = sites.width;
  Grid<int64_t> out(sites.height, sites.width, kNoSite);

  struct Site {
    int64_t index, y, x;
  };
  std::vector<Site> list;
  const auto in = sites.flat();
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (in[i] != 0) {
      const auto s = static_cast<int64_t>(i);
      list.push_back({s, s / w, s % w});
    }
  }
  if (list.empty()) {
    return out;
  }

  // Sites are visited in increasing flat index, so keeping the first strict
  // minimum of d2 picks the smallest (d2, site) pair: the JFA tie-break.
  for (std::size_t i = 0; i < out.size(); ++i) {
    const auto p = static_cast<int64_t>(i);
    const int64_t py = p / w;
    const int64_t px = p % w;
    int64_t best_d2 = std::numeric_limits<int64_t>::max();
    int64_t best_site = kNoSite;
    for (const Site& s : list) {
      const int64_t dy = py - s.y;
      const int64_t dx = px - s.x;
      const int64_t d2 = dy * dy + dx * dx;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_site = s.index;
      }
    }
    out.data()[i] = best_site;
  }
  return out;
}

}  // namespace voronav
