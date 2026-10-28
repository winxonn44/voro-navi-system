#pragma once

// 2D grids in NumPy's layout: row-major, C-contiguous, no padding, so a NumPy
// array and a GridView can share one buffer (see python/bindings.cpp).

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace voronav {

// Non-owning view of an H x W row-major buffer. Cheap to copy; pass by value.
// GridView<const T> is the read-only form and converts from GridView<T>.
template <typename T>
struct GridView {
  T* data = nullptr;
  int height = 0;
  int width = 0;

  GridView() = default;
  GridView(T* data_, int height_, int width_) : data(data_), height(height_), width(width_) {
    if (height_ < 0 || width_ < 0) {
      throw std::invalid_argument("GridView: negative shape");
    }
  }
  template <typename U>
    requires std::is_same_v<const U, T>
  GridView(GridView<U> other)  // implicit, like T* -> const T*
      : data(other.data), height(other.height), width(other.width) {}

  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  }
  // The flat index y * W + x, the same as NumPy's ravel_multi_index.
  [[nodiscard]] std::size_t index(int y, int x) const noexcept {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x);
  }
  [[nodiscard]] T& operator()(int y, int x) const noexcept { return data[index(y, x)]; }
  [[nodiscard]] std::span<T> row(int y) const noexcept {
    return {data + index(y, 0), static_cast<std::size_t>(width)};
  }
  [[nodiscard]] std::span<T> flat() const noexcept { return {data, size()}; }
};

// Owning H x W grid. Moving it keeps the buffer address, so a view (or a NumPy
// array wrapping data()) stays valid across a move.
template <typename T>
class Grid {
 public:
  Grid() = default;
  Grid(int height, int width, T fill = T{})
      : height_(check_dim(height)),
        width_(check_dim(width)),
        data_(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), fill) {}

  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] T* data() noexcept { return data_.data(); }
  [[nodiscard]] const T* data() const noexcept { return data_.data(); }

  [[nodiscard]] GridView<T> view() noexcept { return {data_.data(), height_, width_}; }
  [[nodiscard]] GridView<const T> view() const noexcept { return {data_.data(), height_, width_}; }
  [[nodiscard]] T& operator()(int y, int x) noexcept { return view()(y, x); }
  [[nodiscard]] const T& operator()(int y, int x) const noexcept { return view()(y, x); }

  friend bool operator==(const Grid&, const Grid&) = default;

 private:
  static int check_dim(int n) {
    if (n < 0) {
      throw std::invalid_argument("Grid: negative shape");
    }
    return n;
  }

  int height_ = 0;
  int width_ = 0;
  std::vector<T> data_;
};

// Deep copy of a view into a new owning grid.
template <typename T>
Grid<std::remove_const_t<T>> to_grid(GridView<T> view) {
  Grid<std::remove_const_t<T>> out(view.height, view.width);
  const auto src = view.flat();
  std::copy(src.begin(), src.end(), out.data());
  return out;
}

}  // namespace voronav
