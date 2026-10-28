// C0 stub: proves the Google Benchmark wiring. The JFA benchmarks (A6 in the
// design doc) replace this from C1 on.

#include <benchmark/benchmark.h>

#include <cstdint>

#include "voronav/grid.hpp"

namespace {

void BM_GridFillFlatIndex(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  voronav::Grid<int64_t> g(n, n);
  for (auto _ : state) {
    const auto v = g.view();
    for (int y = 0; y < v.height; ++y) {
      for (int x = 0; x < v.width; ++x) {
        v(y, x) = static_cast<int64_t>(v.index(y, x));
      }
    }
    benchmark::DoNotOptimize(g.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(g.size() * sizeof(int64_t)));
}
BENCHMARK(BM_GridFillFlatIndex)->Arg(256)->Arg(1024)->Arg(2048);

}  // namespace
