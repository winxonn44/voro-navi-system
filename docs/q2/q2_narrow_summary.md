# Q2 summary

20 maps, 1000 queries, 14200 planner runs. Successful paths that failed the independent collision check: 0.

## All maps, raw paths

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 1000/1000 | 1.000 | 4.2 px | 1.00 px | 5.72 ms |
| weighted A* λ=4 | 1000/1000 | 1.030 | 6.4 px | 1.25 px | 13.12 ms |
| weighted A* λ=16 | 1000/1000 | 1.061 | 6.8 px | 1.59 px | 16.08 ms |
| weighted A* λ=64 | 1000/1000 | 1.083 | 7.0 px | 1.71 px | 17.83 ms |
| weighted A* λ=128 | 1000/1000 | 1.089 | 7.1 px | 1.72 px | 18.23 ms |
| Voronoi [nearest] | 1000/1000 | 1.382 | 7.5 px | 1.71 px | 2.30 ms |
| Voronoi [nearest-road16] | 1000/1000 | 1.382 | 7.6 px | 1.71 px | 3.64 ms |
| Voronoi [best16] | 1000/1000 | 1.365 | 7.6 px | 1.66 px | 5.80 ms |
| Voronoi [best64] | 1000/1000 | 1.369 | 7.6 px | 1.70 px | 5.74 ms |
| RRT* | 1/200 | 1.025 | 12.8 px | 5.00 px | 41.24 ms |

## All maps, smoothed (`preserve`: minimum clearance never lowered)

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 1000/1000 | 1.000 | 4.4 px | 1.00 px | 5.72 ms |
| weighted A* λ=4 | 1000/1000 | 1.034 | 5.6 px | 1.25 px | 13.12 ms |
| weighted A* λ=16 | 1000/1000 | 1.055 | 6.1 px | 1.59 px | 16.08 ms |
| weighted A* λ=64 | 1000/1000 | 1.069 | 6.3 px | 1.71 px | 17.83 ms |
| weighted A* λ=128 | 1000/1000 | 1.070 | 6.4 px | 1.72 px | 18.23 ms |
| Voronoi [nearest] | 1000/1000 | 1.351 | 6.5 px | 1.71 px | 2.30 ms |
| Voronoi [nearest-road16] | 1000/1000 | 1.359 | 6.6 px | 1.71 px | 3.64 ms |
| Voronoi [best16] | 1000/1000 | 1.352 | 6.5 px | 1.66 px | 5.80 ms |
| Voronoi [best64] | 1000/1000 | 1.355 | 6.6 px | 1.70 px | 5.74 ms |
| RRT* | 1/200 | 1.024 | 10.9 px | 5.00 px | 41.24 ms |

## Voronoi [best64] vs weighted A* λ=64, paired by query

| map family | queries | Voronoi [best64] min clearance ≥ weighted A* λ=64 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 99% | +0.4 px | 1.021 | 1.9× |
| MovingAI room | 500 | 100% | +0.8 px | 1.886 | 4.1× |
| **all** | 1000 | 99% | +0.6 px | 1.137 | 2.7× |

## Voronoi [nearest-road16] vs weighted A* λ=16, paired by query

| map family | queries | Voronoi [nearest-road16] min clearance ≥ weighted A* λ=16 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 99% | +0.7 px | 1.082 | 3.5× |
| MovingAI room | 500 | 100% | +1.1 px | 1.931 | 4.9× |
| **all** | 1000 | 100% | +0.9 px | 1.222 | 4.0× |

## Voronoi [nearest] vs A*, paired by query

| map family | queries | Voronoi [nearest] min clearance ≥ A* | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 100% | +3.6 px | 1.240 | 3.3× |
| MovingAI room | 500 | 100% | +3.2 px | 2.024 | 1.7× |
| **all** | 1000 | 100% | +3.4 px | 1.382 | 2.5× |

## Roadmap build per map family

| map family | maps | median size | median build | skeleton px (median) | bridges / map | pruned px / map |
|---|---|---|---|---|---|---|
| MovingAI maze | 10 | 512 px | 376.4 ms | 36553 | 30.5 | 13 |
| MovingAI room | 10 | 512 px | 1453.5 ms | 36147 | 865.0 | 80 |
