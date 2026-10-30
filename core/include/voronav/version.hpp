#pragma once

#include <string_view>

namespace voronav {

// The library version, e.g. "0.0.1". Set by CMake from project(VERSION).
std::string_view version() noexcept;

}  // namespace voronav
