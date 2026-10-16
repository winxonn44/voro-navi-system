# Q2 summary

20 maps, 1000 queries, 14200 planner runs. Successful paths that failed the independent collision check: 0.

## All maps, raw paths

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 1000/1000 | 1.000 | 4.2 px | 1.00 px | 5.84 ms |
| weighted A* λ=4 | 1000/1000 | 1.030 | 6.4 px | 1.25 px | 13.46 ms |
| weighted A* λ=16 | 1000/1000 | 1.061 | 6.8 px | 1.59 px | 16.21 ms |
| weighted A* λ=64 | 1000/1000 | 1.083 | 7.0 px | 1.71 px | 18.02 ms |
| weighted A* λ=128 | 1000/1000 | 1.089 | 7.1 px | 1.72 px | 18.49 ms |
| Voronoi [nearest] | 1000/1000 | 1.384 | 7.6 px | 1.73 px | 2.44 ms |
| Voronoi [nearest-road16] | 1000/1000 | 1.385 | 7.7 px | 1.73 px | 3.71 ms |
| Voronoi [best16] | 1000/1000 | 1.370 | 7.6 px | 1.69 px | 5.93 ms |
| Voronoi [best64] | 1000/1000 | 1.373 | 7.7 px | 1.73 px | 5.87 ms |
| RRT* | 1/200 | 1.025 | 12.8 px | 5.00 px | 40.98 ms |

## All maps, smoothed (`preserve`: minimum clearance never lowered)

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 1000/1000 | 1.000 | 4.4 px | 1.00 px | 5.84 ms |
| weighted A* λ=4 | 1000/1000 | 1.034 | 5.6 px | 1.25 px | 13.46 ms |
| weighted A* λ=16 | 1000/1000 | 1.055 | 6.1 px | 1.59 px | 16.21 ms |
| weighted A* λ=64 | 1000/1000 | 1.069 | 6.3 px | 1.71 px | 18.02 ms |
| weighted A* λ=128 | 1000/1000 | 1.070 | 6.4 px | 1.72 px | 18.49 ms |
| Voronoi [nearest] | 1000/1000 | 1.357 | 6.5 px | 1.73 px | 2.44 ms |
| Voronoi [nearest-road16] | 1000/1000 | 1.365 | 6.6 px | 1.73 px | 3.71 ms |
| Voronoi [best16] | 1000/1000 | 1.359 | 6.6 px | 1.69 px | 5.93 ms |
| Voronoi [best64] | 1000/1000 | 1.360 | 6.6 px | 1.73 px | 5.87 ms |
| RRT* | 1/200 | 1.024 | 10.9 px | 5.00 px | 40.98 ms |

## Voronoi [best64] vs weighted A* λ=64, paired by query

| map family | queries | Voronoi [best64] min clearance ≥ weighted A* λ=64 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 100% | +0.4 px | 1.022 | 1.9× |
| MovingAI room | 500 | 100% | +0.9 px | 1.920 | 4.2× |
| **all** | 1000 | 100% | +0.6 px | 1.133 | 2.7× |

## Voronoi [nearest-road16] vs weighted A* λ=16, paired by query

| map family | queries | Voronoi [nearest-road16] min clearance ≥ weighted A* λ=16 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 100% | +0.7 px | 1.082 | 3.4× |
| MovingAI room | 500 | 100% | +1.2 px | 1.992 | 5.0× |
| **all** | 1000 | 100% | +0.9 px | 1.218 | 4.0× |

## Voronoi [nearest] vs A*, paired by query

| map family | queries | Voronoi [nearest] min clearance ≥ A* | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| MovingAI maze | 500 | 100% | +3.6 px | 1.240 | 3.3× |
| MovingAI room | 500 | 100% | +3.2 px | 2.099 | 1.7× |
| **all** | 1000 | 100% | +3.4 px | 1.384 | 2.5× |

## Roadmap build per map family

| map family | maps | median size | median build | skeleton px (median) | bridges / map | pruned px / map |
|---|---|---|---|---|---|---|
| MovingAI maze | 10 | 512 px | 83.9 ms | 36553 | 30.5 | 13 |
| MovingAI room | 10 | 512 px | 91.0 ms | 36155 | 865.0 | 80 |
