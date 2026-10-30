add_executable(core_bench bench_grid.cpp bench_jfa.cpp)
target_link_libraries(core_bench PRIVATE voronav::core voronav_warnings benchmark::benchmark_main)
