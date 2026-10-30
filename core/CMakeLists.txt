cmake_minimum_required(VERSION 3.24)

# Keep in sync with `version` in core/pyproject.toml (tests/test_core.py checks it).
project(voronav_core VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static lib is linked into the Python module

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# A plain `cmake -S core` build gets tests and benchmarks; a pip build
# (scikit-build-core sets SKBUILD) gets only the Python module.
# (SKBUILD is "2" under scikit-build-core, not a boolean, so test it with if().)
if(SKBUILD)
  set(_voronav_standalone OFF)
  set(_voronav_pip ON)
else()
  set(_voronav_standalone ON)
  set(_voronav_pip OFF)
endif()
option(VORONAV_BUILD_TESTS "Build the GoogleTest suite" ${_voronav_standalone})
option(VORONAV_BUILD_BENCH "Build the Google Benchmark suite" ${_voronav_standalone})
option(VORONAV_BUILD_PYTHON "Build the pybind11 module voronav_core._core" ${_voronav_pip})
option(VORONAV_WERROR "Treat warnings as errors (CI)" OFF)
set(VORONAV_SANITIZE "" CACHE STRING
    "Comma-separated -fsanitize= list, e.g. address,undefined or thread")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Sanitizers apply to everything, dependencies included: TSan in particular
# reports false positives when only part of the program is instrumented.
if(VORONAV_SANITIZE)
  add_compile_options(-fsanitize=${VORONAV_SANITIZE} -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${VORONAV_SANITIZE})
endif()

# Warnings apply to our targets only, never to fetched dependencies.
add_library(voronav_warnings INTERFACE)
target_compile_options(voronav_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  $<$<BOOL:${VORONAV_WERROR}>:-Werror>)

# Dependencies, pinned. FetchContent downloads them at configure time.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

if(VORONAV_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

if(VORONAV_BUILD_BENCH)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(benchmark)
endif()

if(VORONAV_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)
endif()

# The library: headers in include/voronav, sources in src.
add_library(voronav_core STATIC src/jfa.cpp src/oracle.cpp src/version.cpp)
add_library(voronav::core ALIAS voronav_core)
target_include_directories(voronav_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(voronav_core PUBLIC cxx_std_20)
target_compile_definitions(voronav_core PRIVATE
  VORONAV_CORE_VERSION="${PROJECT_VERSION}")
target_link_libraries(voronav_core PRIVATE voronav_warnings)

if(VORONAV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VORONAV_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(VORONAV_BUILD_PYTHON)
  pybind11_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE voronav::core voronav_warnings)
  install(TARGETS _core LIBRARY DESTINATION voronav_core)
endif()
