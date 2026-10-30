// First serial number for run_steps (design doc A7 #1). The timed region is
// one run_steps call: the JFA passes plus its copy of init and allocation of
// the second buffer (about one pass's worth of memory traffic). The seed grid
// is built once, outside it. 1% sites, jfa+2 schedule.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"

namespace {

void BM_RunStepsSerial(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  std::mt19937_64 rng(0);
  std::bernoulli_distribution coin(0.01);
  voronav::Grid<uint8_t> sites(n, n, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = static_cast<uint8_t>(coin(rng));
  }
  const voronav::Grid<int64_t> init = voronav::init_nearest(sites.view());
  const std::vector<int> steps = voronav::step_schedule(n, n, voronav::Variant::JfaPlus2);
  for (auto _ : state) {
    voronav::Grid<int64_t> out = voronav::run_steps(init.view(), steps);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(steps.size());
}
BENCHMARK(BM_RunStepsSerial)->Arg(256)->Arg(512)->Unit(benchmark::kMillisecond);

}  // namespace
