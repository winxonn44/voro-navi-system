// Unit tests for thread_pool.hpp: every row is visited exactly once under
// both schedules, the pool survives many reuses and concurrent callers, and
// destroying it leaves no threads behind. Run under TSan in CI.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "voronav/thread_pool.hpp"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#include <sys/qos.h>
#elif defined(__linux__)
#include <filesystem>
#endif

namespace voronav {
namespace {

constexpr std::array<Schedule, 2> kSchedules{Schedule::Static, Schedule::Dynamic};

// Live threads in this process, or -1 where we cannot count them.
int live_threads() {
#if defined(__APPLE__)
  thread_act_array_t threads = nullptr;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) {
    return -1;
  }
  for (mach_msg_type_number_t i = 0; i < count; ++i) {
    mach_port_deallocate(mach_task_self(), threads[i]);
  }
  vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                count * sizeof(thread_act_t));
  return static_cast<int>(count);
#elif defined(__linux__)
  int count = 0;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
    static_cast<void>(entry);
    ++count;
  }
  return count;
#else
  return -1;
#endif
}

// The kernel may finish tearing a joined thread down a moment after join
// returns, so poll briefly until the count is back to at most `limit`. (At
// most: macOS starts and retires system threads, e.g. libdispatch workers,
// on its own schedule, so the count can also drop below the baseline.)
int live_threads_settled(int limit) {
  int n = live_threads();
  for (int i = 0; i < 200 && n > limit; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    n = live_threads();
  }
  return n;
}

// Calls parallel_rows and returns how many times each row was visited, plus
// every (begin, end) range handed out.
struct Visits {
  std::vector<int> per_row;
  std::vector<std::pair<int, int>> ranges;
};

Visits visit(ThreadPool& pool, int rows, Schedule schedule, int chunk) {
  const auto n = static_cast<std::size_t>(rows > 0 ? rows : 0);
  Visits v{std::vector<int>(n, 0), std::vector<std::pair<int, int>>(n)};
  // fn must not throw, so ranges are stored into preallocated slots (there
  // are at most `rows` non-empty calls) through an atomic cursor.
  std::atomic<std::size_t> n_ranges{0};
  auto fn = [&v, &n_ranges](int begin, int end) noexcept {
    for (int r = begin; r < end; ++r) {
      ++v.per_row[static_cast<std::size_t>(r)];  // disjoint rows: no lock needed
    }
    v.ranges[n_ranges.fetch_add(1, std::memory_order_relaxed)] = {begin, end};
  };
  pool.parallel_rows(rows, schedule, chunk, fn);
  v.ranges.resize(n_ranges.load(std::memory_order_relaxed));  // ordered by the pool's barrier
  return v;
}

TEST(ThreadPool, RejectsZeroThreads) { EXPECT_THROW(ThreadPool(0), std::invalid_argument); }

TEST(ThreadPool, RejectsChunkBelowOne) {
  ThreadPool pool(3);
  auto fn = [](int, int) noexcept {};
  EXPECT_THROW(pool.parallel_rows(10, Schedule::Dynamic, 0, fn), std::invalid_argument);
  EXPECT_THROW(pool.parallel_rows(10, Schedule::Static, -4, fn), std::invalid_argument);
  pool.parallel_rows(10, Schedule::Dynamic, 1, fn);  // still usable afterwards
}

TEST(ThreadPool, SizeCountsTheCaller) {
  EXPECT_EQ(ThreadPool(1).size(), 1U);
  EXPECT_EQ(ThreadPool(7).size(), 7U);
}

TEST(ThreadPool, EveryRowExactlyOnce) {
  for (const unsigned n : {1U, 2U, 3U, 7U, 10U}) {
    ThreadPool pool(n);
    for (const Schedule s : kSchedules) {
      for (const int chunk : {1, 3, 4, 16, 1000}) {
        for (const int rows : {1, 2, 5, 7, 9, 10, 11, 64, 100, 1001}) {
          const Visits v = visit(pool, rows, s, chunk);
          for (int r = 0; r < rows; ++r) {
            ASSERT_EQ(v.per_row[static_cast<std::size_t>(r)], 1)
                << "n=" << n << " chunk=" << chunk << " rows=" << rows << " row=" << r;
          }
          for (const auto& [b, e] : v.ranges) {
            ASSERT_LT(b, e);  // never an empty call
            ASSERT_GE(b, 0);
            ASSERT_LE(e, rows);
          }
        }
      }
    }
  }
}

TEST(ThreadPool, ZeroRowsCallsNothing) {
  ThreadPool pool(4);
  std::atomic<int> calls{0};
  auto fn = [&](int, int) noexcept { calls.fetch_add(1, std::memory_order_relaxed); };
  for (const Schedule s : kSchedules) {
    pool.parallel_rows(0, s, 4, fn);
    pool.parallel_rows(-3, s, 4, fn);
  }
  EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPool, ChunkLargerThanRowsIsOneCall) {
  ThreadPool pool(6);
  const Visits v = visit(pool, 5, Schedule::Dynamic, 64);
  ASSERT_EQ(v.ranges.size(), 1U);
  EXPECT_EQ(v.ranges[0], std::make_pair(0, 5));
}

TEST(ThreadPool, StaticGivesEqualContiguousBlocks) {
  // Participant i gets [rows * i / n, rows * (i + 1) / n), like Numba prange.
  for (const unsigned n : {2U, 3U, 7U, 10U}) {
    ThreadPool pool(n);
    for (const int rows : {3, 7, 10, 23, 100}) {
      Visits v = visit(pool, rows, Schedule::Static, 1);
      std::sort(v.ranges.begin(), v.ranges.end());
      std::vector<std::pair<int, int>> want;
      for (unsigned i = 0; i < n; ++i) {
        const int b = static_cast<int>(int64_t{rows} * i / n);
        const int e = static_cast<int>(int64_t{rows} * (i + 1) / n);
        if (b < e) {
          want.emplace_back(b, e);
        }
      }
      EXPECT_EQ(v.ranges, want) << "n=" << n << " rows=" << rows;
    }
  }
}

TEST(ThreadPool, DynamicClaimsAlignedChunks) {
  ThreadPool pool(7);
  for (const int chunk : {1, 4, 16}) {
    Visits v = visit(pool, 103, Schedule::Dynamic, chunk);
    std::sort(v.ranges.begin(), v.ranges.end());
    int expect_begin = 0;
    for (const auto& [b, e] : v.ranges) {
      EXPECT_EQ(b, expect_begin);
      EXPECT_EQ(e, std::min(b + chunk, 103));
      expect_begin = e;
    }
    EXPECT_EQ(expect_begin, 103);
  }
}

TEST(ThreadPool, ReusedAcrossManyCalls) {
  // Thousands of back-to-back jobs on one pool: catches lost wake-ups and
  // generation mix-ups that a handful of calls would not.
  ThreadPool pool(5);
  std::vector<int64_t> out(37, 0);
  for (int iter = 0; iter < 5000; ++iter) {
    const Schedule s = kSchedules[static_cast<std::size_t>(iter % 2)];
    auto fn = [&out, iter](int b, int e) noexcept {
      for (int r = b; r < e; ++r) {
        out[static_cast<std::size_t>(r)] += iter + r;
      }
    };
    pool.parallel_rows(37, s, 1 + iter % 5, fn);
  }
  for (int r = 0; r < 37; ++r) {
    EXPECT_EQ(out[static_cast<std::size_t>(r)], int64_t{4999} * 5000 / 2 + int64_t{5000} * r);
  }
}

TEST(ThreadPool, PassesSeeThePreviousPassesWrites) {
  // Each job reads rows other threads wrote in the previous job: the
  // completion barrier must order them (TSan flags it otherwise).
  ThreadPool pool(4);
  constexpr int kRows = 64;
  std::vector<int64_t> a(kRows, 1);
  std::vector<int64_t> b(kRows, 0);
  for (int pass = 0; pass < 200; ++pass) {
    auto fn = [&a, &b](int lo, int hi) noexcept {
      for (int r = lo; r < hi; ++r) {
        b[static_cast<std::size_t>(r)] = a[static_cast<std::size_t>((r + 17) % kRows)] + 1;
      }
    };
    pool.parallel_rows(kRows, pass % 2 == 0 ? Schedule::Dynamic : Schedule::Static, 3, fn);
    std::swap(a, b);
  }
  for (const int64_t v : a) {
    EXPECT_EQ(v, 201);
  }
}

TEST(ThreadPool, ConcurrentCallersAreSerialized) {
  ThreadPool pool(4);
  constexpr int kRows = 50;
  std::vector<std::vector<int>> results(3, std::vector<int>(kRows, 0));
  std::vector<std::thread> callers;
  callers.reserve(results.size());
  for (auto& res : results) {
    callers.emplace_back([&pool, &res] {
      for (int iter = 0; iter < 200; ++iter) {
        auto fn = [&res](int b, int e) noexcept {
          for (int r = b; r < e; ++r) {
            ++res[static_cast<std::size_t>(r)];
          }
        };
        pool.parallel_rows(kRows, Schedule::Dynamic, 2, fn);
      }
    });
  }
  for (auto& t : callers) {
    t.join();
  }
  for (const auto& res : results) {
    for (const int v : res) {
      EXPECT_EQ(v, 200);
    }
  }
}

TEST(ThreadPool, UsedImmediatelyAfterConstruction) {
  // The first job can be published before a worker reaches its wait; it must
  // still run (workers start from generation 0, not from a fresh load).
  for (int iter = 0; iter < 200; ++iter) {
    ThreadPool pool(4);
    const Visits v = visit(pool, 8, Schedule::Static, 1);
    for (const int c : v.per_row) {
      ASSERT_EQ(c, 1);
    }
  }
}

TEST(ThreadPool, DestructionJoinsEveryWorker) {
  // Sanitizer runtimes (TSan) start a helper thread when the first thread is
  // created; start and join one first so it is part of the baseline.
  std::thread([] {}).join();
  const int before = live_threads_settled(live_threads());
  if (before < 0) {
    GTEST_SKIP() << "cannot count threads on this platform";
  }
  int during = 0;
  {
    ThreadPool pool(10);
    during = live_threads();
    const Visits v = visit(pool, 100, Schedule::Dynamic, 4);
    EXPECT_EQ(v.per_row.size(), 100U);
  }
  const int after = live_threads_settled(before);
  EXPECT_LE(after, before);
  EXPECT_GE(during - after, 9);  // 9 workers existed, then were joined; the caller is the 10th
  // Many short-lived pools, some never used: none may leak a thread.
  for (int iter = 0; iter < 100; ++iter) {
    ThreadPool pool(3);
    if (iter % 2 == 0) {
      static_cast<void>(visit(pool, 9, Schedule::Static, 1));
    }
  }
  EXPECT_LE(live_threads_settled(before), before);
}

TEST(ThreadPool, UserInteractiveQosReachesTheWorkersOnly) {
  ThreadPool pool(3, WorkerQos::UserInteractive);
  EXPECT_EQ(pool.qos(), WorkerQos::UserInteractive);
  EXPECT_EQ(ThreadPool(2).qos(), WorkerQos::Default);
  // Static, 3 rows over 3 participants: row i runs on participant i, and
  // participant 0 is this thread.
  std::array<std::thread::id, 3> who{};
  [[maybe_unused]] std::array<int, 3> qos{};
  auto fn = [&who, &qos](int begin, int end) noexcept {
    for (int r = begin; r < end; ++r) {
      who[static_cast<std::size_t>(r)] = std::this_thread::get_id();
#if defined(__APPLE__)
      qos_class_t c = QOS_CLASS_UNSPECIFIED;
      static_cast<void>(pthread_get_qos_class_np(pthread_self(), &c, nullptr));
      qos[static_cast<std::size_t>(r)] = static_cast<int>(c);
#endif
    }
  };
  pool.parallel_rows(3, Schedule::Static, 1, fn);
  EXPECT_EQ(who[0], std::this_thread::get_id());
  EXPECT_NE(who[1], who[0]);
  EXPECT_NE(who[2], who[0]);
#if defined(__APPLE__)
  EXPECT_EQ(qos[1], static_cast<int>(QOS_CLASS_USER_INTERACTIVE));
  EXPECT_EQ(qos[2], static_cast<int>(QOS_CLASS_USER_INTERACTIVE));
  // The caller keeps whatever class it had (a terminal-launched main
  // thread is often already USER_INTERACTIVE; std::thread workers start at
  // DEFAULT).
  qos_class_t mine = QOS_CLASS_UNSPECIFIED;
  static_cast<void>(pthread_get_qos_class_np(pthread_self(), &mine, nullptr));
  EXPECT_EQ(qos[0], static_cast<int>(mine));
#endif
}

}  // namespace
}  // namespace voronav
