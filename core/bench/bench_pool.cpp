// Pool vs spawning threads every pass (design doc A4: "Spawning threads per
// pass costs tens of microseconds ... Measure it to prove it").
//
//   BM_EmptyPass{Pool,Spawn}/threads   one pass whose work is a no-op: the
//                                      fixed cost of starting and finishing
//                                      a parallel pass.
//   BM_Jfa{Pool,Spawn}/n/threads       a full jfa+2 run_steps at n x n, 1%
//                                      sites, static schedule, the same
//                                      kernel (jfa_pass_rows) either way.
// Per-pass overhead of spawning = (Spawn - Pool) / passes. Wall-clock time
// (UseRealTime), since the work runs on several threads.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "voronav/grid.hpp"
#include "voronav/jfa.hpp"
#include "voronav/thread_pool.hpp"

namespace {

using voronav::Grid;
using voronav::GridView;

// The benchmark-only alternative to ThreadPool: create threads - 1
// std::jthreads for this one pass, run participant 0 on the caller, join.
// Static blocks, exactly as ThreadPool's Static schedule.
template <typename Fn>
void spawn_rows(unsigned threads, int rows, const Fn& fn) {
  const auto share = [&fn, threads, rows](unsigned i) {
    const int64_t b = int64_t{rows} * i / threads;
    const int64_t e = int64_t{rows} * (i + 1) / threads;
    if (b < e) {
      fn(static_cast<int>(b), static_cast<int>(e));
    }
  };
  std::vector<std::jthread> workers;
  workers.reserve(threads - 1);
  for (unsigned i = 1; i < threads; ++i) {
    workers.emplace_back(share, i);
  }
  share(0);
}  // ~jthread joins

void BM_EmptyPassPool(benchmark::State& state) {
  const auto threads = static_cast<unsigned>(state.range(0));
  voronav::ThreadPool pool(threads);
  auto noop = [](int b, int e) noexcept { benchmark::DoNotOptimize(b + e); };
  for (auto _ : state) {
    pool.parallel_rows(static_cast<int>(threads), voronav::Schedule::Static, 1, noop);
  }
}
BENCHMARK(BM_EmptyPassPool)->DenseRange(2, 10, 2)->UseRealTime()->Unit(benchmark::kMicrosecond);

void BM_EmptyPassSpawn(benchmark::State& state) {
  const auto threads = static_cast<unsigned>(state.range(0));
  auto noop = [](int b, int e) noexcept { benchmark::DoNotOptimize(b + e); };
  for (auto _ : state) {
    spawn_rows(threads, static_cast<int>(threads), noop);
  }
}
BENCHMARK(BM_EmptyPassSpawn)->DenseRange(2, 10, 2)->UseRealTime()->Unit(benchmark::kMicrosecond);

struct JfaInput {
  Grid<int64_t> init;
  std::vector<int> steps;
};

JfaInput make_input(int n) {
  std::mt19937_64 rng(0);
  std::bernoulli_distribution coin(0.01);
  Grid<uint8_t> sites(n, n, 0);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    sites.data()[i] = static_cast<uint8_t>(coin(rng));
  }
  return {voronav::init_nearest(sites.view()),
          voronav::step_schedule(n, n, voronav::Variant::JfaPlus2)};
}

void BM_JfaPool(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  voronav::ThreadPool pool(static_cast<unsigned>(state.range(1)));
  const JfaInput in = make_input(n);
  for (auto _ : state) {
    Grid<int64_t> out =
        voronav::run_steps(in.init.view(), in.steps, &pool, voronav::Schedule::Static);
    benchmark::DoNotOptimize(out.data());
  }
  state.counters["passes"] = static_cast<double>(in.steps.size());
}

// run_steps with the pool swapped for spawn_rows; everything else (copy of
// init, second buffer, swap) is the same.
void BM_JfaSpawn(benchmark::State& state) {
  const auto n = static_cast<int>(state.range(0));
  const auto threads = static_cast<unsigned>(state.range(1));
  const JfaInput in = make_input(n);
  for (auto _ : state) {
    Grid<int64_t> cur = voronav::to_grid(in.init.view());
    Grid<int64_t> nxt(n, n);
    for (const int k : in.steps) {
      const GridView<const int64_t> src = cur.view();
      const GridView<int64_t> dst = nxt.view();
      spawn_rows(threads, n,
                 [src, dst, k](int y0, int y1) { voronav::jfa_pass_rows(src, dst, k, y0, y1); });
      std::swap(cur, nxt);
    }
    benchmark::DoNotOptimize(cur.data());
  }
  state.counters["passes"] = static_cast<double>(in.steps.size());
}

void jfa_args(benchmark::internal::Benchmark* b) {
  for (const int n : {256, 1024}) {
    for (const int t : {4, 10}) {
      b->Args({n, t});
    }
  }
}
BENCHMARK(BM_JfaPool)->Apply(jfa_args)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_JfaSpawn)->Apply(jfa_args)->UseRealTime()->Unit(benchmark::kMillisecond);

}  // namespace
