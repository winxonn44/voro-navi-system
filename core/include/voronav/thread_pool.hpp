#pragma once

// A fixed pool of worker threads for row-parallel passes (design doc A4).
//
// The pool is created once and reused for every pass: starting a pass costs
// one atomic increment plus a wake-up per sleeping worker, instead of
// creating and joining threads (measured in docs/CORE.md).
//
// Participants: ThreadPool(n) spawns n - 1 std::thread workers; the thread
// that calls parallel_rows is participant 0 and works too, so n threads
// compute and ThreadPool(1) runs everything on the caller.
//
// Schedules, as in the Numba backend (src/voronav/voronoi/jfa_numba.py):
//   Static:  participant i takes the contiguous block
//            [rows * i / n, rows * (i + 1) / n), like Numba's prange.
//   Dynamic: every participant repeatedly claims the next `chunk` rows from
//            a shared atomic counter (fetch_add) until none are left, so a
//            fast core simply takes more rows than a slow one.
// Neither schedule affects results when fn writes only its own rows, which
// is what makes the JFA output independent of threads, schedule and chunk.
//
// Synchronization (all C++20 std::atomic, no OpenMP, no busy-waiting):
//   start: the caller writes the job (fn, rows, schedule, chunk), resets the
//     row counter and the pending count, then bumps generation_ with a
//     release RMW and notify_all. Workers block in generation_.wait() and
//     read it with acquire, so the job fields happen-before their use.
//   rows: the dynamic counter uses relaxed fetch_add. It only hands out
//     disjoint row ranges (atomicity guarantees each value is returned
//     once); no data is published through it.
//   finish: each worker does pending_.fetch_sub(1, release) after its last
//     write; the caller waits until an acquire load of pending_ reads 0.
//     The fetch_subs form one release sequence, so every worker's writes to
//     the output buffer happen-before the caller returns, and therefore
//     before run_steps swaps the buffers and starts the next pass.
//   stop: ~ThreadPool sets stopping_ and bumps generation_ the same way, so
//     a woken worker sees the flag, returns, and is joined.
//   idle: workers sleep in atomic::wait (futex on Linux, ulock on macOS).
//     The library may spin briefly before sleeping; an idle pool uses no CPU.
//
// False sharing: the three shared atomics sit on separate 128-byte lines
// (Apple M-series L1 lines are 128 bytes; x86 and most Arm Linux use 64), so
// workers claiming rows do not invalidate the line the sleepers wait on.
// Worker state other than these is local to each thread.
//
// parallel_rows is serialized by an internal mutex: two threads may share
// one pool (for example two Python threads with the GIL released), and each
// call then runs alone. fn must not throw (checked at compile time).
//
// Plain std::thread, not std::jthread / std::stop_token: Xcode 15's libc++
// (GitHub's macos-14 runners) does not provide those two, and the pool needs
// its own wake-up for stopping anyway (a stop_token cannot wake an
// atomic::wait).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace voronav {

enum class Schedule : std::uint8_t { Static, Dynamic };

// Scheduling class for the pool's worker threads (design doc A7 #6). macOS
// cannot pin a thread to a core, but a QoS class steers it: Default leaves
// the workers as std::thread creates them; UserInteractive calls
// pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) in each worker
// before its first job, the class macOS prefers to run on performance cores.
// Elsewhere UserInteractive is accepted and does nothing. The calling thread
// (participant 0) is never changed. Results do not depend on it.
enum class WorkerQos : std::uint8_t { Default, UserInteractive };

class ThreadPool {
 public:
  // n_threads >= 1 participants, the caller included (n_threads - 1 workers
  // are spawned here). Throws std::invalid_argument for 0.
  explicit ThreadPool(unsigned n_threads, WorkerQos qos = WorkerQos::Default);
  // Stops and joins every worker. Must not run concurrently with
  // parallel_rows.
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Number of participants (workers + the calling thread).
  [[nodiscard]] unsigned size() const noexcept { return n_threads_; }
  [[nodiscard]] WorkerQos qos() const noexcept { return qos_; }

  // Calls fn(begin, end) on disjoint row ranges covering [0, rows) exactly
  // once, spread over the participants by `schedule`, and returns when all
  // of them are done. `chunk` (>= 1) is the dynamic claim size and is
  // ignored by Static. rows <= 0 returns at once. Throws
  // std::invalid_argument if chunk < 1.
  template <typename Fn>
  void parallel_rows(int rows, Schedule schedule, int chunk, Fn& fn) {
    static_assert(std::is_nothrow_invocable_v<Fn&, int, int>,
                  "parallel_rows: fn(begin, end) must be noexcept");
    run(rows, schedule, chunk, &call<Fn>, &fn);
  }

 private:
  using RowFn = void (*)(void* ctx, int begin, int end) noexcept;

  template <typename Fn>
  static void call(void* ctx, int begin, int end) noexcept {
    (*static_cast<Fn*>(ctx))(begin, end);
  }

  void run(int rows, Schedule schedule, int chunk, RowFn fn, void* ctx);
  void worker_loop(unsigned index) noexcept;
  void do_share(unsigned index) noexcept;

  static constexpr std::size_t kLine = 128;

  unsigned n_threads_;
  WorkerQos qos_;
  std::mutex run_mutex_;  // one parallel_rows at a time

  // The current job. Plain fields: written by the caller before the release
  // bump of generation_, read by workers after the acquire load.
  RowFn fn_ = nullptr;
  void* ctx_ = nullptr;
  int rows_ = 0;
  int chunk_ = 1;
  Schedule schedule_ = Schedule::Dynamic;
  bool stopping_ = false;  // set by ~ThreadPool, published like the job fields

  alignas(kLine) std::atomic<std::uint32_t> generation_{0};  // bumped per job and on stop
  alignas(kLine) std::atomic<std::int64_t> next_row_{0};     // dynamic: next unclaimed row
  alignas(kLine) std::atomic<std::int32_t> pending_{0};      // workers still running the job

  // Joined in ~ThreadPool, before any member above is destroyed.
  alignas(kLine) std::vector<std::thread> workers_;
};

}  // namespace voronav
