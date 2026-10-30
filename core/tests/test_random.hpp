#pragma once

// Portable random draws for the tests. std::mt19937_64 produces the same bits
// on every platform, but std::uniform_int_distribution and
// std::bernoulli_distribution are implementation-defined: libc++ (macOS) and
// libstdc++ (Linux) turn the same bits into different numbers, so a test using
// them checks different grids on each platform. These helpers depend only on
// the generator's output, so every platform tests the same inputs.

#include <cstdint>
#include <random>

namespace voronav::test {

// Uniform in [0, n), n > 0, by rejection: discard the lowest 2^64 mod n raw
// values so the remaining range is a multiple of n (as arc4random_uniform).
inline uint64_t uniform_below(std::mt19937_64& rng, uint64_t n) {
  const uint64_t threshold = (0 - n) % n;  // == 2^64 mod n
  for (;;) {
    const uint64_t r = rng();
    if (r >= threshold) {
      return r % n;
    }
  }
}

// Uniform in [lo, hi], lo <= hi.
inline int uniform_int(std::mt19937_64& rng, int lo, int hi) {
  const auto span = static_cast<uint64_t>(static_cast<int64_t>(hi) - lo + 1);
  return static_cast<int>(lo + static_cast<int64_t>(uniform_below(rng, span)));
}

// True with probability p: a uniform double in [0, 1) from the top 53 bits.
inline bool bernoulli(std::mt19937_64& rng, double p) {
  return static_cast<double>(rng() >> 11U) * 0x1.0p-53 < p;
}

}  // namespace voronav::test
