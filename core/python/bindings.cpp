// pybind11 module voronav_core._core.
//
// Arrays cross the boundary without copies in either direction:
//   in:  py::array_t<int64_t, c_style | forcecast> is wrapped as a GridView
//        over NumPy's own buffer (a copy happens only if the caller passes a
//        non-int64 or non-contiguous array; functions that write in place
//        refuse such arrays instead of silently writing to a temporary);
//   out: an owning Grid is moved to the heap and handed to NumPy with a
//        capsule that frees it, so the result array uses the Grid's buffer.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <climits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "voronav/grid.hpp"
#include "voronav/version.hpp"

namespace py = pybind11;

namespace {

using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

int checked_dim(py::ssize_t n) {
  if (n > INT_MAX) {
    throw std::invalid_argument("array dimension " + std::to_string(n) + " exceeds INT_MAX");
  }
  return static_cast<int>(n);
}

void check_2d_contiguous(const Int64Array& a) {
  if (a.ndim() != 2) {
    throw std::invalid_argument("expected a 2D array, got ndim=" + std::to_string(a.ndim()));
  }
  // c_style guarantees this after conversion; assert it so a future change to
  // the array type cannot quietly turn the view into a strided read.
  if ((a.flags() & py::array::c_style) == 0) {
    throw std::invalid_argument("expected a C-contiguous array");
  }
}

voronav::GridView<const int64_t> view_of(const Int64Array& a) {
  check_2d_contiguous(a);
  return {a.data(), checked_dim(a.shape(0)), checked_dim(a.shape(1))};
}

voronav::GridView<int64_t> mutable_view_of(Int64Array& a) {
  check_2d_contiguous(a);
  return {a.mutable_data(), checked_dim(a.shape(0)), checked_dim(a.shape(1))};
}

py::array_t<int64_t> to_numpy(voronav::Grid<int64_t>&& grid) {
  auto owner = std::make_unique<voronav::Grid<int64_t>>(std::move(grid));
  const py::ssize_t h = owner->height();
  const py::ssize_t w = owner->width();
  int64_t* data = owner->data();
  const py::capsule free_grid(owner.get(),
                              [](void* p) { delete static_cast<voronav::Grid<int64_t>*>(p); });
  owner.release();  // NOLINT(bugprone-unused-return-value): the capsule owns it now
  const auto item = static_cast<py::ssize_t>(sizeof(int64_t));
  return {{h, w}, {w * item, item}, data, free_grid};
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "voronav C++ core (C0 scaffolding: version, Grid round-trip).";

  m.def(
      "version", [] { return std::string(voronav::version()); },
      "Version of the compiled C++ library.");

  m.def(
      "data_address",
      [](const Int64Array& a) { return reinterpret_cast<std::uintptr_t>(view_of(a).data); },
      py::arg("a").noconvert(),
      "Address of the buffer C++ sees. Equals the NumPy array's own address "
      "when the array crossed without a copy.");

  m.def(
      "fill_flat_index",
      [](Int64Array& a) {
        const auto v = mutable_view_of(a);
        for (int y = 0; y < v.height; ++y) {
          for (int x = 0; x < v.width; ++x) {
            v(y, x) = static_cast<int64_t>(v.index(y, x));
          }
        }
      },
      py::arg("a").noconvert(),
      "Write y * W + x into a C-contiguous int64 2D array, in place. Other "
      "arrays raise TypeError rather than being copied.");

  m.def(
      "roundtrip", [](const Int64Array& a) { return to_numpy(voronav::to_grid(view_of(a))); },
      py::arg("a"),
      "Copy a 2D array into a C++ Grid and return it as a new NumPy array "
      "that owns the Grid's buffer.");
}
