// run_steps, serial (design doc A7 #1) and on the thread pool (C3), and the
// C4 kernels side by side (A7 #2-#4). The timed region is one run_steps
// call: the JFA passes plus its copy of init and allocation of the second
// buffer (about one pass's worth of memory traffic; for Kernel::Packed, the
// conversion to and from the packed layout instead). The seed grid is built
// once, outside it.
//
//   BM_RunStepsSerial/n                         1% sites, jfa+2, Auto kernel
//   BM_RunStepsPool/n/threads/schedule/chunk    schedule 0 = static, 1 = dynamic
//   BM_JfaKernel/variant/density/kernel/threads/schedule/chunk/n
//       variant: 1 = jfa+1, 2 = jfa+2; density: sites per 1000 pixels;
//       kernel: 1 = Reference, 2 = Split, 3 = Packed (jfa.hpp Kernel);
//       threads: 1 = serial (no pool), else a pool of that many;
//       schedule 0 = static, 1 = dynamic. n is last so CI's smoke filter
//       '/256$' picks up the 256^2 cases. scripts/bench_core_kernels.py
//       drives this one with cool-downs and saves the raw results.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

namespace {

voronav::Grid<int64_t> seed_grid(int n, double density = 0.01) {
  std::mt19937_64 rng(0);
  std::bernoulli_distribution coin(density);
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

// One seed grid per (n, density) for the whole process, so every kernel,
// thread count and schedule times the same input.
const voronav::Grid<int64_t>& cached_seed(int n, int per_mille) {
  static auto* const cache = new std::map<std::pair<int, int>, voronav::Grid<int64_t>>;
  const auto key = std::make_pair(n, per_mille);
  auto it = cache->find(key);
  if (it == cache->end()) {
    it = cache->emplace(key, seed_grid(n, per_mille / 1000.0)).first;
  }
  return it->second;
}

void BM_JfaKernel(benchmark::State& state) {
  const auto variant =
      state.range(0) == 1 ? voronav::Variant::JfaPlus1 : voronav::Variant::JfaPlus2;
  const auto per_mille = static_cast<int>(state.range(1));
  const auto kernel = static_cast<voronav::Kernel>(state.range(2));
  const auto threads = static_cast<unsigned>(state.range(3));
  const auto schedule =
      state.range(4) == 0 ? voronav::Schedule::Static : voronav::Schedule::Dynamic;
  const auto chunk = static_cast<int>(state.range(5));
  const auto n = static_cast<int>(state.range(6));
  std::unique_ptr<voronav::ThreadPool> pool;
  if (threads > 1) {
    pool = std::make_unique<voronav::ThreadPool>(threads);
  }
  const voronav::Grid<int64_t>& init = cached_seed(n, per_mille);
  const std::vector<int> steps = voronav::step_schedule(n, n, variant);
  for (auto _ : state) {
    voronav::Grid<int64_t> out =
        voronav::run_steps(init.view(), steps, pool.get(), schedule, chunk, kernel);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(steps.size());
  state.counters["Mpx_per_s"] =
      benchmark::Counter(static_cast<double>(n) * n * static_cast<double>(steps.size()) / 1e6,
                         benchmark::Counter::kIsIterationInvariantRate);
}
// Kernels x {serial, 4, 10 threads} (dynamic, chunk 4) on the A6 inputs; the
// chunk sweep (A7 #4): both schedules x chunk {1..32} at 4 and 10 threads;
// thread scaling 1-10 at 2048^2 for both schedules; and 4096^2. Duplicates
// across these sets are registered once. 256^2 entries keep CI's smoke run
// short.
void kernel_args(benchmark::internal::Benchmark* b) {
  using Row = std::vector<int64_t>;
  std::set<Row> rows;
  const auto product = [&rows](const std::vector<std::vector<int64_t>>& axes) {
    std::vector<std::size_t> idx(axes.size(), 0);
    for (;;) {
      Row r;
      for (std::size_t i = 0; i < axes.size(); ++i) {
        r.push_back(axes[i][idx[i]]);
      }
      rows.insert(r);
      std::size_t i = 0;
      while (i < axes.size() && ++idx[i] == axes[i].size()) {
        idx[i++] = 0;
      }
      if (i == axes.size()) {
        return;
      }
    }
  };
  product({{1, 2}, {1, 10}, {1, 2, 3}, {1, 4, 10}, {1}, {4}, {256, 1024, 2048}});
  product({{1}, {1}, {3}, {4, 10}, {0, 1}, {1, 2, 4, 8, 16, 32}, {1024, 2048}});
  product({{1}, {1}, {1, 3}, {1, 2, 4, 6, 8, 10}, {0, 1}, {4}, {2048}});
  product({{1, 2}, {1, 10}, {3}, {1, 10}, {1}, {4}, {4096}});
  for (const Row& r : rows) {
    b->Args(r);
  }
}
BENCHMARK(BM_JfaKernel)->Apply(kernel_args)->UseRealTime()->Unit(benchmark::kMillisecond);

// A7 #6: worker QoS. BM_JfaQos/qos/schedule/threads/n, qos 0 = Default,
// 1 = UserInteractive; Packed kernel, jfa+1, 0.1% sites, chunk 4.
void BM_JfaQos(benchmark::State& state) {
  const auto qos =
      state.range(0) == 0 ? voronav::WorkerQos::Default : voronav::WorkerQos::UserInteractive;
  const auto schedule =
      state.range(1) == 0 ? voronav::Schedule::Static : voronav::Schedule::Dynamic;
  const auto threads = static_cast<unsigned>(state.range(2));
  const auto n = static_cast<int>(state.range(3));
  voronav::ThreadPool pool(threads, qos);
  const voronav::Grid<int64_t>& init = cached_seed(n, 1);
  const std::vector<int> steps = voronav::step_schedule(n, n, voronav::Variant::JfaPlus1);
  for (auto _ : state) {
    voronav::Grid<int64_t> out = voronav::run_steps(
        init.view(), steps, &pool, schedule, voronav::kDefaultChunk, voronav::Kernel::Packed);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(steps.size());
}
BENCHMARK(BM_JfaQos)
    ->ArgsProduct({{0, 1}, {0, 1}, {4, 6, 8, 10}, {1024, 2048}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
