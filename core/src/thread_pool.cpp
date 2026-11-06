#include "voronav/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace voronav {

namespace {

void apply_qos(WorkerQos qos) noexcept {
#if defined(__APPLE__)
  if (qos == WorkerQos::UserInteractive) {
    // Fails only for an invalid class; nothing to recover, the worker just
    // keeps its default class.
    static_cast<void>(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0));
  }
#else
  static_cast<void>(qos);
#endif
}

}  // namespace

ThreadPool::ThreadPool(unsigned n_threads, WorkerQos qos) : n_threads_(n_threads), qos_(qos) {
  if (n_threads == 0) {
    throw std::invalid_argument("ThreadPool: n_threads must be >= 1");
  }
  workers_.reserve(n_threads - 1);
  for (unsigned i = 1; i < n_threads; ++i) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
}

ThreadPool::~ThreadPool() {
  // Published by the release bump: a worker's acquire load of the new
  // generation happens-after this store, so it sees stopping_ == true.
  stopping_ = true;
  generation_.fetch_add(1, std::memory_order_release);
  generation_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
}

void ThreadPool::run(int rows, Schedule schedule, int chunk, RowFn fn, void* ctx) {
  if (chunk < 1) {
    throw std::invalid_argument("parallel_rows: chunk must be >= 1");
  }
  if (rows <= 0) {
    return;
  }
  const std::scoped_lock lock(run_mutex_);
  fn_ = fn;
  ctx_ = ctx;
  rows_ = rows;
  chunk_ = chunk;
  schedule_ = schedule;
  next_row_.store(0, std::memory_order_relaxed);
  if (workers_.empty()) {
    do_share(0);
    return;
  }
  pending_.store(static_cast<std::int32_t>(workers_.size()), std::memory_order_relaxed);
  // Publishes the job fields and the two stores above to the workers.
  generation_.fetch_add(1, std::memory_order_release);
  generation_.notify_all();

  do_share(0);

  // Acquire pairs with every worker's release fetch_sub: their output writes
  // happen-before we return.
  for (std::int32_t left = pending_.load(std::memory_order_acquire); left != 0;
       left = pending_.load(std::memory_order_acquire)) {
    pending_.wait(left, std::memory_order_acquire);
  }
}

void ThreadPool::worker_loop(unsigned index) noexcept {
  // Not generation_.load(): the caller may already have published the first
  // job before this thread got here, and loading would skip it. Every worker
  // starts from the constructor's value, 0.
  std::uint32_t seen = 0;
  apply_qos(qos_);  // qos_ is set before the thread starts
  for (;;) {
    generation_.wait(seen, std::memory_order_acquire);  // returns once it differs
    seen = generation_.load(std::memory_order_acquire);
    if (stopping_) {
      return;
    }
    do_share(index);
    // Release: our writes to the output happen-before the caller's acquire
    // load that reads 0. Only the last worker needs to wake the caller.
    if (pending_.fetch_sub(1, std::memory_order_release) == 1) {
      pending_.notify_one();
    }
  }
}

void ThreadPool::do_share(unsigned index) noexcept {
  const auto rows = static_cast<std::int64_t>(rows_);
  if (schedule_ == Schedule::Static) {
    const auto n = static_cast<std::int64_t>(n_threads_);
    const auto i = static_cast<std::int64_t>(index);
    const std::int64_t begin = rows * i / n;
    const std::int64_t end = rows * (i + 1) / n;
    if (begin < end) {
      fn_(ctx_, static_cast<int>(begin), static_cast<int>(end));
    }
    return;
  }
  const auto chunk = static_cast<std::int64_t>(chunk_);
  for (;;) {
    // Relaxed: the counter only partitions rows; nothing is published by it.
    const std::int64_t begin = next_row_.fetch_add(chunk, std::memory_order_relaxed);
    if (begin >= rows) {
      return;
    }
    fn_(ctx_, static_cast<int>(begin), static_cast<int>(std::min(begin + chunk, rows)));
  }
}

}  // namespace voronav
