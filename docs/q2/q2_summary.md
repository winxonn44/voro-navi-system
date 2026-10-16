# Q2 summary

94 maps, 4100 queries, 58220 planner runs. Successful paths that failed the independent collision check: 0.

## All maps, raw paths

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 4100/4100 | 1.000 | 16.6 px | 2.93 px | 1.12 ms |
| weighted A* λ=4 | 4100/4100 | 1.028 | 22.7 px | 5.78 px | 3.25 ms |
| weighted A* λ=16 | 4100/4100 | 1.067 | 24.6 px | 6.98 px | 4.36 ms |
| weighted A* λ=64 | 4100/4100 | 1.106 | 26.5 px | 7.29 px | 5.28 ms |
| weighted A* λ=128 | 4100/4100 | 1.118 | 27.2 px | 7.35 px | 5.63 ms |
| Voronoi [nearest] | 4100/4100 | 1.249 | 28.3 px | 6.80 px | 0.78 ms |
| Voronoi [nearest-road16] | 4100/4100 | 1.261 | 29.0 px | 7.12 px | 1.02 ms |
| Voronoi [best16] | 4100/4100 | 1.162 | 27.6 px | 7.03 px | 3.77 ms |
| Voronoi [best64] | 4100/4100 | 1.178 | 28.4 px | 7.24 px | 3.96 ms |
| RRT* | 656/820 | 0.964 | 20.8 px | 3.31 px | 184.41 ms |

## All maps, smoothed (`preserve`: minimum clearance never lowered)

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 4100/4100 | 1.000 | 17.1 px | 3.01 px | 1.12 ms |
| weighted A* λ=4 | 4100/4100 | 1.044 | 20.9 px | 5.80 px | 3.25 ms |
| weighted A* λ=16 | 4100/4100 | 1.074 | 22.5 px | 6.99 px | 4.36 ms |
| weighted A* λ=64 | 4100/4100 | 1.098 | 23.8 px | 7.31 px | 5.28 ms |
| weighted A* λ=128 | 4100/4100 | 1.105 | 24.3 px | 7.36 px | 5.63 ms |
| Voronoi [nearest] | 4100/4100 | 1.133 | 24.2 px | 6.83 px | 0.78 ms |
| Voronoi [nearest-road16] | 4100/4100 | 1.145 | 24.9 px | 7.15 px | 1.02 ms |
| Voronoi [best16] | 4100/4100 | 1.112 | 24.1 px | 7.05 px | 3.77 ms |
| Voronoi [best64] | 4100/4100 | 1.127 | 24.6 px | 7.26 px | 3.96 ms |
| RRT* | 656/820 | 1.000 | 20.6 px | 3.38 px | 184.41 ms |

## Voronoi [best64] vs weighted A* λ=64, paired by query

| map family | queries | Voronoi [best64] min clearance ≥ weighted A* λ=64 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 98% | +8.5 px | 1.083 | 2.4× |
| random 128² | 250 | 99% | +0.5 px | 1.045 | 2.4× |
| random 256² | 250 | 99% | +1.4 px | 1.060 | 2.7× |
| random 512² | 250 | 98% | +3.3 px | 1.069 | 2.7× |
| corridor / two rooms | 200 | 98% | +3.0 px | 1.031 | 0.9× |
| HouseExpo | 1000 | 97% | +2.0 px | 1.065 | 0.7× |
| MovingAI bg512 | 500 | 99% | +2.0 px | 1.073 | 1.5× |
| MovingAI dao | 500 | 95% | -0.7 px | 1.032 | 1.3× |
| MovingAI maze | 400 | 100% | +0.5 px | 1.032 | 1.4× |
| MovingAI street | 500 | 88% | +1.2 px | 1.071 | 3.1× |
| **all** | 4100 | 97% | +1.8 px | 1.058 | 1.5× |

## Voronoi [nearest-road16] vs weighted A* λ=16, paired by query

| map family | queries | Voronoi [nearest-road16] min clearance ≥ weighted A* λ=16 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 97% | +18.4 px | 1.238 | 5.3× |
| random 128² | 250 | 97% | +1.1 px | 1.124 | 3.3× |
| random 256² | 250 | 98% | +3.1 px | 1.169 | 5.8× |
| random 512² | 250 | 93% | +7.3 px | 1.200 | 5.0× |
| corridor / two rooms | 200 | 100% | +8.3 px | 1.244 | 4.2× |
| HouseExpo | 1000 | 99% | +5.1 px | 1.244 | 4.9× |
| MovingAI bg512 | 500 | 90% | +3.3 px | 1.144 | 3.0× |
| MovingAI dao | 500 | 94% | -0.2 px | 1.064 | 1.9× |
| MovingAI maze | 400 | 100% | +0.9 px | 1.100 | 2.4× |
| MovingAI street | 500 | 80% | +3.4 px | 1.183 | 4.1× |
| **all** | 4100 | 94% | +4.4 px | 1.164 | 3.6× |

## Voronoi [nearest] vs A*, paired by query

| map family | queries | Voronoi [nearest] min clearance ≥ A* | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 100% | +40.6 px | 1.295 | 2.6× |
| random 128² | 250 | 100% | +3.9 px | 1.216 | 1.2× |
| random 256² | 250 | 100% | +8.6 px | 1.247 | 2.0× |
| random 512² | 250 | 99% | +18.7 px | 1.281 | 1.8× |
| corridor / two rooms | 200 | 100% | +15.1 px | 1.279 | 1.2× |
| HouseExpo | 1000 | 100% | +12.4 px | 1.324 | 1.3× |
| MovingAI bg512 | 500 | 99% | +11.5 px | 1.225 | 1.4× |
| MovingAI dao | 500 | 99% | +2.7 px | 1.122 | 1.4× |
| MovingAI maze | 400 | 100% | +3.9 px | 1.192 | 2.5× |
| MovingAI street | 500 | 99% | +12.2 px | 1.256 | 1.4× |
| **all** | 4100 | 99% | +11.7 px | 1.249 | 1.5× |

## Roadmap build per map family

| map family | maps | median size | median build | skeleton px (median) | bridges / map | pruned px / map |
|---|---|---|---|---|---|---|
| random 1024² | 5 | 1024 px | 260.6 ms | 14822 | 9.0 | 34 |
| random 128² | 5 | 128 px | 10.2 ms | 1484 | 1.6 | 10 |
| random 256² | 5 | 256 px | 30.9 ms | 3403 | 4.0 | 25 |
| random 512² | 5 | 512 px | 66.6 ms | 7203 | 6.2 | 23 |
| corridor / two rooms | 4 | 328 px | 38.1 ms | 1930 | 1.5 | 0 |
| HouseExpo | 20 | 311 px | 38.6 ms | 2408 | 6.7 | 3 |
| MovingAI bg512 | 10 | 512 px | 51.8 ms | 6433 | 61.5 | 426 |
| MovingAI dao | 10 | 271 px | 25.6 ms | 4073 | 15.3 | 134 |
| MovingAI maze | 10 | 512 px | 55.1 ms | 34933 | 8.0 | 13 |
| MovingAI street | 10 | 512 px | 83.1 ms | 12810 | 25.2 | 233 |
