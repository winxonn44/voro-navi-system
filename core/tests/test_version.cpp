#include <gtest/gtest.h>

#include "voronav/version.hpp"

TEST(Version, MatchesCMakeProjectVersion) {
  EXPECT_EQ(voronav::version(), VORONAV_CORE_TEST_EXPECTED_VERSION);
}
