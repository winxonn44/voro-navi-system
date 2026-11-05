// run_steps, serial (design doc A7 #1) and on the thread pool (C3; the
// starting point for C4's schedule and chunk experiments, A7 #4). The timed
// region is one run_steps call: the JFA passes plus its copy of init and
// allocation of the second buffer (about one pass's worth of memory
// traffic). The seed grid is built once, outside it. 1% sites, jfa+2.
//
//   BM_RunStepsSerial/n
//   BM_RunStepsPool/n/threads/schedule/chunk   schedule 0 = static, 1 = dynamic

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

namespace {

voronav::Grid<int64_t> seed_grid(int n) {
  std::mt19937_64 rng(0);
  std::bernoulli_distribution coin(0.01);
  voronav::Grid<uint8_t> sites(n, n, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = static_cast<uint8_t>(coin(rng));
  }
  return voronav::init_nearest(sites.view());
}

void BM_RunStepsSerial(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  const voronav::Grid<int64_t> init = seed_grid(n);
  const std::vector<int> steps = voronav::step_schedule(n, n, voronav::Variant::JfaPlus2);
  for (auto _ : state) {
    voronav::Grid<int64_t> out = voronav::run_steps(init.view(), steps);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(steps.size());
}
BENCHMARK(BM_RunStepsSerial)->Arg(256)->Arg(512)->Unit(benchmark::kMillisecond);

void BM_RunStepsPool(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  voronav::ThreadPool pool(static_cast<unsigned>(state.range(1)));
  const auto schedule =
      state.range(2) == 0 ? voronav::Schedule::Static : voronav::Schedule::Dynamic;
  const auto chunk = static_cast<int>(state.range(3));
  const voronav::Grid<int64_t> init = seed_grid(n);
  const std::vector<int> steps = voronav::step_schedule(n, n, voronav::Variant::JfaPlus2);
  for (auto _ : state) {
    voronav::Grid<int64_t> out = voronav::run_steps(init.view(), steps, &pool, schedule, chunk);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(steps.size());
}
BENCHMARK(BM_RunStepsPool)
    ->ArgsProduct({{256, 512}, {4, 10}, {0, 1}, {4}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
