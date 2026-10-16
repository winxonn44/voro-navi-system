# Q2 summary

94 maps, 4100 queries, 58220 planner runs. Successful paths that failed the independent collision check: 0.

## All maps, raw paths

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 4100/4100 | 1.000 | 16.6 px | 2.93 px | 1.08 ms |
| weighted A* λ=4 | 4100/4100 | 1.028 | 22.7 px | 5.78 px | 3.14 ms |
| weighted A* λ=16 | 4100/4100 | 1.067 | 24.6 px | 6.98 px | 4.26 ms |
| weighted A* λ=64 | 4100/4100 | 1.106 | 26.5 px | 7.29 px | 5.18 ms |
| weighted A* λ=128 | 4100/4100 | 1.118 | 27.2 px | 7.35 px | 5.59 ms |
| Voronoi [nearest] | 4100/4100 | 1.253 | 28.1 px | 6.60 px | 0.66 ms |
| Voronoi [nearest-road16] | 4100/4100 | 1.265 | 28.8 px | 6.92 px | 0.89 ms |
| Voronoi [best16] | 4100/4100 | 1.160 | 27.4 px | 6.98 px | 3.45 ms |
| Voronoi [best64] | 4100/4100 | 1.177 | 28.2 px | 7.19 px | 3.63 ms |
| RRT* | 656/820 | 0.964 | 20.8 px | 3.31 px | 184.22 ms |

## All maps, smoothed (`preserve`: minimum clearance never lowered)

| planner | solved | length / A* (median) | mean clearance | min clearance | median query |
|---|---|---|---|---|---|
| A* | 4100/4100 | 1.000 | 17.1 px | 3.01 px | 1.08 ms |
| weighted A* λ=4 | 4100/4100 | 1.044 | 20.9 px | 5.80 px | 3.14 ms |
| weighted A* λ=16 | 4100/4100 | 1.074 | 22.5 px | 6.99 px | 4.26 ms |
| weighted A* λ=64 | 4100/4100 | 1.098 | 23.8 px | 7.31 px | 5.18 ms |
| weighted A* λ=128 | 4100/4100 | 1.105 | 24.3 px | 7.36 px | 5.59 ms |
| Voronoi [nearest] | 4100/4100 | 1.135 | 24.0 px | 6.66 px | 0.66 ms |
| Voronoi [nearest-road16] | 4100/4100 | 1.146 | 24.7 px | 6.99 px | 0.89 ms |
| Voronoi [best16] | 4100/4100 | 1.112 | 24.0 px | 7.00 px | 3.45 ms |
| Voronoi [best64] | 4100/4100 | 1.126 | 24.5 px | 7.22 px | 3.63 ms |
| RRT* | 656/820 | 1.000 | 20.6 px | 3.38 px | 184.22 ms |

## Voronoi [best64] vs weighted A* λ=64, paired by query

| map family | queries | Voronoi [best64] min clearance ≥ weighted A* λ=64 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 98% | +8.4 px | 1.081 | 2.4× |
| random 128² | 250 | 99% | +0.5 px | 1.045 | 2.2× |
| random 256² | 250 | 99% | +1.4 px | 1.061 | 2.7× |
| random 512² | 250 | 98% | +3.3 px | 1.069 | 2.7× |
| corridor / two rooms | 200 | 98% | +3.0 px | 1.032 | 0.9× |
| HouseExpo | 1000 | 96% | +1.7 px | 1.060 | 0.7× |
| MovingAI bg512 | 500 | 93% | +1.6 px | 1.072 | 1.9× |
| MovingAI dao | 500 | 94% | -0.7 px | 1.033 | 1.4× |
| MovingAI maze | 400 | 99% | +0.5 px | 1.032 | 1.4× |
| MovingAI street | 500 | 86% | +1.1 px | 1.068 | 3.3× |
| **all** | 4100 | 95% | +1.7 px | 1.057 | 1.7× |

## Voronoi [nearest-road16] vs weighted A* λ=16, paired by query

| map family | queries | Voronoi [nearest-road16] min clearance ≥ weighted A* λ=16 | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 97% | +18.4 px | 1.240 | 5.3× |
| random 128² | 250 | 98% | +1.1 px | 1.125 | 3.3× |
| random 256² | 250 | 98% | +3.1 px | 1.169 | 5.8× |
| random 512² | 250 | 93% | +7.3 px | 1.199 | 5.0× |
| corridor / two rooms | 200 | 100% | +8.3 px | 1.244 | 4.5× |
| HouseExpo | 1000 | 96% | +4.9 px | 1.247 | 5.4× |
| MovingAI bg512 | 500 | 79% | +2.7 px | 1.157 | 3.2× |
| MovingAI dao | 500 | 93% | -0.3 px | 1.064 | 2.1× |
| MovingAI maze | 400 | 99% | +0.9 px | 1.100 | 2.5× |
| MovingAI street | 500 | 74% | +2.8 px | 1.182 | 4.3× |
| **all** | 4100 | 91% | +4.1 px | 1.165 | 3.8× |

## Voronoi [nearest] vs A*, paired by query

| map family | queries | Voronoi [nearest] min clearance ≥ A* | mean clearance a − b | length a / b (median) | speedup b_time / a_time (median) |
|---|---|---|---|---|---|
| random 1024² | 250 | 100% | +40.5 px | 1.297 | 2.3× |
| random 128² | 250 | 100% | +3.9 px | 1.216 | 1.1× |
| random 256² | 250 | 100% | +8.6 px | 1.247 | 1.9× |
| random 512² | 250 | 99% | +18.7 px | 1.281 | 1.8× |
| corridor / two rooms | 200 | 100% | +15.1 px | 1.280 | 1.0× |
| HouseExpo | 1000 | 98% | +12.1 px | 1.330 | 1.6× |
| MovingAI bg512 | 500 | 97% | +10.9 px | 1.241 | 1.4× |
| MovingAI dao | 500 | 99% | +2.7 px | 1.123 | 1.5× |
| MovingAI maze | 400 | 100% | +3.9 px | 1.192 | 2.6× |
| MovingAI street | 500 | 98% | +11.5 px | 1.259 | 1.5× |
| **all** | 4100 | 99% | +11.5 px | 1.253 | 1.6× |

## Roadmap build per map family

| map family | maps | median size | median build | skeleton px (median) | bridges / map | pruned px / map |
|---|---|---|---|---|---|---|
| random 1024² | 5 | 1024 px | 482.6 ms | 14818 | 9.0 | 34 |
| random 128² | 5 | 128 px | 9.8 ms | 1484 | 1.6 | 10 |
| random 256² | 5 | 256 px | 30.2 ms | 3407 | 4.0 | 25 |
| random 512² | 5 | 512 px | 99.1 ms | 7203 | 6.2 | 23 |
| corridor / two rooms | 4 | 328 px | 22.0 ms | 1930 | 1.5 | 0 |
| HouseExpo | 20 | 311 px | 42.7 ms | 2403 | 6.7 | 3 |
| MovingAI bg512 | 10 | 512 px | 219.7 ms | 6288 | 61.5 | 426 |
| MovingAI dao | 10 | 271 px | 46.4 ms | 4072 | 15.3 | 134 |
| MovingAI maze | 10 | 512 px | 48.5 ms | 34933 | 8.0 | 13 |
| MovingAI street | 10 | 512 px | 334.8 ms | 12765 | 25.2 | 233 |
