#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "voronav/grid.hpp"

namespace voronav {
namespace {

TEST(Grid, ShapeAndFill) {
  const Grid<int64_t> g(3, 5, -1);
  EXPECT_EQ(g.height(), 3);
  EXPECT_EQ(g.width(), 5);
  EXPECT_EQ(g.size(), 15U);
  for (int y = 0; y < g.height(); ++y) {
    for (int x = 0; x < g.width(); ++x) {
      EXPECT_EQ(g(y, x), -1);
    }
  }
}

TEST(Grid, RowMajorFlatIndexMatchesNumPy) {
  // nearest[y, x] holds y * W + x in the Python code; the view must agree.
  Grid<int64_t> g(4, 7);
  const GridView<int64_t> v = g.view();
  for (int y = 0; y < v.height; ++y) {
    for (int x = 0; x < v.width; ++x) {
      v(y, x) = static_cast<int64_t>(v.index(y, x));
    }
  }
  for (std::size_t i = 0; i < g.size(); ++i) {
    EXPECT_EQ(g.data()[i], static_cast<int64_t>(i));
  }
  EXPECT_EQ(v.row(2).front(), 14);
  EXPECT_EQ(v.row(2).size(), 7U);
}

TEST(Grid, EmptyAndDegenerateShapes) {
  const Grid<int64_t> empty;
  EXPECT_EQ(empty.size(), 0U);
  const Grid<int64_t> one_row(1, 9);
  EXPECT_EQ(one_row.view().flat().size(), 9U);
  const Grid<int64_t> zero_wide(5, 0);
  EXPECT_EQ(zero_wide.size(), 0U);
}

TEST(Grid, RejectsNegativeShape) {
  EXPECT_THROW(Grid<int64_t>(-1, 3), std::invalid_argument);
  int64_t dummy = 0;
  EXPECT_THROW(GridView<int64_t>(&dummy, 1, -1), std::invalid_argument);
}

TEST(Grid, MoveKeepsBufferAddress) {
  // The bindings hand data() to NumPy and then move the grid; this must hold.
  Grid<int64_t> a(8, 8, 3);
  const int64_t* before = a.data();
  const Grid<int64_t> b = std::move(a);
  EXPECT_EQ(b.data(), before);
}

TEST(GridView, ConstViewAliasesMutableView) {
  Grid<int64_t> g(2, 2, 0);
  const GridView<int64_t> mut = g.view();
  const GridView<const int64_t> ro = mut;
  mut(1, 1) = 42;
  EXPECT_EQ(ro(1, 1), 42);
  EXPECT_EQ(ro.data, g.data());
}

TEST(GridView, ToGridIsADeepCopy) {
  Grid<int64_t> g(3, 3, 1);
  Grid<int64_t> copy = to_grid(GridView<const int64_t>(g.view()));
  EXPECT_EQ(copy, g);
  EXPECT_NE(copy.data(), g.data());
  copy(0, 0) = 9;
  EXPECT_EQ(g(0, 0), 1);
}

}  // namespace
}  // namespace voronav
