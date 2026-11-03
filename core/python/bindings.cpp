// pybind11 module voronav_core._core.
//
// Arrays cross the boundary without copies in either direction:
//   in:  the JFA functions take any object and accept it only if it already
//        is a C-contiguous, aligned 2D NumPy array of the right dtype (int64
//        for nearest grids; bool or uint8 for sites). It is then viewed in
//        place as a GridView over NumPy's own buffer. Anything else raises
//        ValueError (std::invalid_argument) instead of being converted, so a
//        silent copy cannot happen;
//   out: an owning Grid is moved to the heap and handed to NumPy with a
//        capsule that frees it, so the result array uses the Grid's buffer.
// The JFA functions release the GIL while they compute; the input arrays stay
// alive because the caller holds references to them for the whole call.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/oracle.hpp"
#include "voronav/version.hpp"

namespace py = pybind11;

namespace {

using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using SitesArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

int checked_dim(py::ssize_t n) {
  if (n > INT_MAX) {
    throw std::invalid_argument("array dimension " + std::to_string(n) + " exceeds INT_MAX");
  }
  return static_cast<int>(n);
}

void check_2d_contiguous(const py::array& a) {
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

// The strict checks behind zero-copy input: shape, layout, alignment and an
// exact dtype, each reported as ValueError naming what to pass instead.
void check_strict(const py::array& a, const char* who) {
  check_2d_contiguous(a);
  const auto address = reinterpret_cast<std::uintptr_t>(a.data());
  if (a.size() > 0 && address % static_cast<std::uintptr_t>(a.itemsize()) != 0) {
    throw std::invalid_argument(std::string(who) + ": expected an aligned array");
  }
}

py::array as_array(const py::handle& obj, const char* who) {
  if (!py::isinstance<py::array>(obj)) {
    throw std::invalid_argument(std::string(who) + ": expected a NumPy array");
  }
  return py::reinterpret_borrow<py::array>(obj);
}

// nearest grids: int64, viewed in place.
voronav::GridView<const int64_t> strict_int64_view(const py::handle& obj, const char* who) {
  const py::array a = as_array(obj, who);
  if (!py::isinstance<py::array_t<int64_t>>(a)) {
    throw std::invalid_argument(std::string(who) + ": expected dtype int64, got " +
                                py::str(a.dtype()).cast<std::string>());
  }
  check_strict(a, who);
  const auto typed = py::reinterpret_borrow<Int64Array>(a);  // no conversion: dtype checked
  return {typed.data(), checked_dim(a.shape(0)), checked_dim(a.shape(1))};
}

// sites masks: bool or uint8, both one byte per cell, viewed in place as
// uint8 (NumPy stores bool as 0 or 1). Any nonzero byte is a site.
voronav::GridView<const uint8_t> strict_sites_view(const py::handle& obj, const char* who) {
  const py::array a = as_array(obj, who);
  if (!py::isinstance<py::array_t<bool>>(a) && !py::isinstance<py::array_t<uint8_t>>(a)) {
    throw std::invalid_argument(std::string(who) + ": expected dtype bool or uint8, got " +
                                py::str(a.dtype()).cast<std::string>());
  }
  check_strict(a, who);
  const auto typed = py::reinterpret_borrow<SitesArray>(a);  // no conversion: 1-byte dtype checked
  return {typed.data(), checked_dim(a.shape(0)), checked_dim(a.shape(1))};
}

voronav::Variant parse_variant(const std::string& name) {
  if (name == "jfa") {
    return voronav::Variant::Jfa;
  }
  if (name == "jfa+1") {
    return voronav::Variant::JfaPlus1;
  }
  if (name == "jfa+2") {
    return voronav::Variant::JfaPlus2;
  }
  if (name == "1+jfa") {
    return voronav::Variant::OnePlusJfa;
  }
  throw std::invalid_argument("variant must be one of ('jfa', 'jfa+1', 'jfa+2', '1+jfa'), got '" +
                              name + "'");
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

// Runs fn without the GIL and returns its Grid as a NumPy array.
template <typename Fn>
py::array_t<int64_t> compute(Fn&& fn) {
  voronav::Grid<int64_t> out;
  {
    const py::gil_scoped_release release;
    out = std::forward<Fn>(fn)();
  }
  return to_numpy(std::move(out));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "voronav C++ core: serial JFA, the brute-force oracle, Grid round-trip.";

  m.def(
      "version", [] { return std::string(voronav::version()); },
      "Version of the compiled C++ library.");

  m.def(
      "data_address",
      [](const py::object& a) {
        const py::array arr = as_array(a, "data_address");
        if (arr.itemsize() == 1) {
          return reinterpret_cast<std::uintptr_t>(strict_sites_view(a, "data_address").data);
        }
        return reinterpret_cast<std::uintptr_t>(strict_int64_view(a, "data_address").data);
      },
      py::arg("a"),
      "Address of the buffer C++ sees, through the same checks as the JFA "
      "functions (int64, or bool/uint8 for sites). Equals the NumPy array's "
      "own address when the array crossed without a copy.");

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

  m.def(
      "init_nearest",
      [](const py::object& sites) {
        const auto v = strict_sites_view(sites, "init_nearest");
        return compute([v] { return voronav::init_nearest(v); });
      },
      py::arg("sites"),
      "Seed grid (int64 [H, W]): each site points at its own flat index, "
      "every other cell is -1. sites: C-contiguous bool or uint8 [H, W].");

  m.def(
      "step_schedule",
      [](int height, int width, const std::string& variant) {
        return voronav::step_schedule(height, width, parse_variant(variant));
      },
      py::arg("height"), py::arg("width"), py::arg("variant") = "jfa",
      "JFA step lengths for an H x W grid; variant is 'jfa', 'jfa+1', "
      "'jfa+2' or '1+jfa', as in voronav.voronoi.step_schedule.");

  m.def(
      "run_steps",
      [](const py::object& init, const std::vector<int>& steps) {
        const auto v = strict_int64_view(init, "run_steps");
        return compute([v, &steps] { return voronav::run_steps(v, steps); });
      },
      py::arg("init"), py::arg("steps"),
      "Serial JFA: one double-buffered pass per step, bit-identical to "
      "jfa_numpy.run_steps. init: C-contiguous int64 [H, W] of -1 or flat "
      "site indices. Returns a new int64 [H, W] array.");

  m.def(
      "distance_sq",
      [](const py::object& nearest) {
        const auto v = strict_int64_view(nearest, "distance_sq");
        return compute([v] { return voronav::distance_sq(v); });
      },
      py::arg("nearest"),
      "Squared distance from each pixel to its assigned site, -1 where "
      "unassigned (int64 [H, W]).");

  m.def(
      "brute_force",
      [](const py::object& sites) {
        const auto v = strict_sites_view(sites, "brute_force");
        return compute([v] { return voronav::brute_force(v); });
      },
      py::arg("sites"),
      "Exact nearest-site grid by brute force (the oracle), with the same "
      "(d2, site) tie-break as JFA. sites: C-contiguous bool or uint8 [H, W].");
}
