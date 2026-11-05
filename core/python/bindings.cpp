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
//
// Threads: run_steps(..., threads=n) runs on a voronav::ThreadPool of n
// threads that is created on first use and cached in this module for the
// life of the process, one per thread count, so passes and calls reuse the
// same workers. The cache is only touched with the GIL held, which
// serializes it; the pool itself serializes concurrent run_steps calls.
// After fork() the child has none of the parent's worker threads, so the
// child forgets (leaks) the cached pools and builds new ones on demand.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/oracle.hpp"
#include "voronav/thread_pool.hpp"
#include "voronav/version.hpp"

#include <pthread.h>

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

voronav::Schedule parse_schedule(const std::string& name) {
  if (name == "static") {
    return voronav::Schedule::Static;
  }
  if (name == "dynamic") {
    return voronav::Schedule::Dynamic;
  }
  throw std::invalid_argument("schedule must be 'static' or 'dynamic', got '" + name + "'");
}

constexpr int kMaxThreads = 1024;

unsigned hardware_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : n;
}

// threads=None means every hardware thread, as in the Numba backend.
unsigned resolve_threads(const std::optional<int>& threads) {
  if (!threads) {
    return hardware_threads();
  }
  if (*threads < 1 || *threads > kMaxThreads) {
    throw std::invalid_argument("threads must be in [1, " + std::to_string(kMaxThreads) +
                                "] or None, got " + std::to_string(*threads));
  }
  return static_cast<unsigned>(*threads);
}

using PoolCache = std::map<unsigned, std::unique_ptr<voronav::ThreadPool>>;
PoolCache* g_pools = nullptr;  // never destroyed: workers may outlive interpreter teardown

// The cached pool for n threads (GIL held). n == 1 is the serial path and
// needs no pool.
voronav::ThreadPool* pool_for(unsigned n) {
  if (n == 1) {
    return nullptr;
  }
  if (g_pools == nullptr) {
    g_pools = new PoolCache;
  }
  auto& slot = (*g_pools)[n];
  if (!slot) {
    slot = std::make_unique<voronav::ThreadPool>(n);
  }
  return slot.get();
}

// pthread_atfork child handler: the child has only the forking thread, so
// the cached pools' workers are gone and joining them would hang. Drop the
// cache without running any destructor.
void forget_pools_after_fork() { g_pools = nullptr; }

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
  m.doc() = "voronav C++ core: JFA (serial or on a thread pool), the brute-force oracle.";
  pthread_atfork(nullptr, nullptr, &forget_pools_after_fork);

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
      [](const py::object& init, const std::vector<int>& steps, const std::optional<int>& threads,
         const std::string& schedule, int chunk) {
        const auto v = strict_int64_view(init, "run_steps");
        const voronav::Schedule sch = parse_schedule(schedule);
        voronav::ThreadPool* pool = pool_for(resolve_threads(threads));  // GIL held
        return compute([v, &steps, pool, sch, chunk] {
          return voronav::run_steps(v, steps, pool, sch, chunk);
        });
      },
      py::arg("init"), py::arg("steps"), py::arg("threads") = 1, py::arg("schedule") = "dynamic",
      py::arg("chunk") = voronav::kDefaultChunk,
      "JFA: one double-buffered pass per step, bit-identical to "
      "jfa_numpy.run_steps for every threads/schedule/chunk. init: "
      "C-contiguous int64 [H, W] of -1 or flat site indices. threads: 1 "
      "(serial, the default), n > 1 (a cached pool of n threads), or None "
      "(all hardware threads). schedule: 'dynamic' (rows claimed `chunk` at "
      "a time from a shared counter) or 'static' (equal contiguous blocks). "
      "Returns a new int64 [H, W] array.");

  m.def("hardware_threads", &hardware_threads,
        "std::thread::hardware_concurrency() (at least 1): what threads=None uses.");

  m.def(
      "_pool_sizes",
      [] {
        std::vector<unsigned> sizes;
        if (g_pools != nullptr) {
          for (const auto& [n, pool] : *g_pools) {
            sizes.push_back(n);
          }
        }
        return sizes;
      },
      "Thread counts with a cached pool (for tests: pools are reused, not "
      "created per call).");

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
