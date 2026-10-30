#pragma once

// Brute-force exact discrete Voronoi diagram: the ground truth for JFA. A port
// of src/voronav/voronoi/oracle.py.

#include <cstdint>

#include "voronav/grid.hpp"

namespace voronav {

// For every pixel, the site (any nonzero cell) with the smallest (d2, site)
// pair, the same tie-break as JFA (see jfa.hpp); kNoSite everywhere if there
// are no sites. O(pixels x sites).
//
// Compare JFA to it by distance, not by site: on tied pixels JFA may never
// have seen the smallest-index site and keeps another equidistant one, which
// is still a correct assignment.
Grid<int64_t> brute_force(GridView<const uint8_t> sites);

}  // namespace voronav
