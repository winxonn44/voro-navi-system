include(GoogleTest)

add_executable(core_tests test_grid.cpp test_jfa.cpp test_oracle.cpp test_version.cpp)
target_link_libraries(core_tests PRIVATE voronav::core voronav_warnings GTest::gtest_main)
target_compile_definitions(core_tests PRIVATE
  VORONAV_CORE_TEST_EXPECTED_VERSION="${PROJECT_VERSION}")
gtest_discover_tests(core_tests DISCOVERY_MODE PRE_TEST)
