# Q3 summary

Part A: 19 maps, 39900 planner runs. Part B: 10 maps, 150000 per-frame plans.

## Part A, cluttered maps: collision % (path enters a true obstacle)

| corruption | level | A* | wA* λ16 | wA* λ64 | Vor nearest | Vor best64 | stranded % (all planners) |
|---|---|---|---|---|---|---|---|
| boundary_shift | -3 | 0 | 0 | 0 | 0 | 0 | 28 |
| boundary_shift | -2 | 0 | 0 | 0 | 0 | 0 | 18 |
| boundary_shift | -1 | 0 | 0 | 0 | 0 | 0 | 5 |
| boundary_shift | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_shift | 1 | 35 | 48 | 50 | 38 | 42 | 0 |
| boundary_shift | 2 | 87 | 55 | 53 | 68 | 63 | 0 |
| boundary_shift | 3 | 92 | 87 | 83 | 88 | 85 | 0 |
| boundary_shift | 4 | 92 | 90 | 83 | 88 | 87 | 0 |
| region_dropout | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| region_dropout | 0.1 | 0 | 0 | 0 | 0 | 0 | 18 |
| region_dropout | 0.2 | 0 | 0 | 0 | 0 | 0 | 53 |
| region_dropout | 0.3 | 0 | 0 | 0 | 0 | 0 | 82 |
| region_dropout | 0.5 | 0 | 0 | 0 | 0 | 0 | 98 |
| thin_obstacle_miss | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 4 | 30 | 45 | 45 | 35 | 38 | 0 |
| thin_obstacle_miss | 6 | 37 | 67 | 58 | 65 | 62 | 0 |
| boundary_noise | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_noise | 1 | 18 | 18 | 12 | 30 | 17 | 3 |
| boundary_noise | 2 | 33 | 13 | 12 | 32 | 10 | 15 |
| boundary_noise | 3 | 65 | 33 | 35 | 55 | 35 | 12 |

## Part A, cluttered maps: margin violation % (true clearance < robot radius)

| corruption | level | A* | wA* λ16 | wA* λ64 | Vor nearest | Vor best64 |
|---|---|---|---|---|---|---|
| boundary_shift | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_shift | 1 | 83 | 63 | 62 | 75 | 63 |
| boundary_shift | 2 | 90 | 77 | 75 | 92 | 82 |
| boundary_shift | 3 | 92 | 97 | 92 | 92 | 90 |
| boundary_shift | 4 | 92 | 97 | 92 | 92 | 90 |
| thin_obstacle_miss | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 2 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 4 | 47 | 62 | 62 | 73 | 62 |
| thin_obstacle_miss | 6 | 50 | 77 | 77 | 92 | 85 |
| boundary_noise | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_noise | 1 | 47 | 23 | 17 | 40 | 28 |
| boundary_noise | 2 | 40 | 17 | 12 | 45 | 25 |
| boundary_noise | 3 | 72 | 43 | 40 | 58 | 42 |

## Part A, structural maps: collision % (path enters a true obstacle)

| corruption | level | A* | wA* λ16 | wA* λ64 | Vor nearest | Vor best64 | stranded % (all planners) |
|---|---|---|---|---|---|---|---|
| boundary_shift | -3 | 0 | 0 | 0 | 0 | 0 | 38 |
| boundary_shift | -2 | 0 | 0 | 0 | 0 | 0 | 26 |
| boundary_shift | -1 | 0 | 0 | 0 | 0 | 0 | 12 |
| boundary_shift | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_shift | 1 | 30 | 19 | 18 | 20 | 19 | 0 |
| boundary_shift | 2 | 78 | 37 | 36 | 43 | 40 | 0 |
| boundary_shift | 3 | 81 | 45 | 43 | 51 | 47 | 0 |
| boundary_shift | 4 | 82 | 47 | 44 | 53 | 49 | 0 |
| region_dropout | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| region_dropout | 0.1 | 0 | 0 | 0 | 0 | 0 | 29 |
| region_dropout | 0.2 | 0 | 0 | 0 | 0 | 0 | 63 |
| region_dropout | 0.3 | 0 | 0 | 0 | 0 | 0 | 74 |
| region_dropout | 0.5 | 0 | 0 | 0 | 0 | 0 | 96 |
| thin_obstacle_miss | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 4 | 3 | 3 | 3 | 5 | 5 | 0 |
| thin_obstacle_miss | 6 | 11 | 13 | 12 | 14 | 12 | 0 |
| boundary_noise | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_noise | 1 | 25 | 11 | 9 | 18 | 10 | 4 |
| boundary_noise | 2 | 42 | 15 | 12 | 25 | 17 | 15 |
| boundary_noise | 3 | 47 | 20 | 19 | 32 | 22 | 10 |

## Part A, structural maps: margin violation % (true clearance < robot radius)

| corruption | level | A* | wA* λ16 | wA* λ64 | Vor nearest | Vor best64 |
|---|---|---|---|---|---|---|
| boundary_shift | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_shift | 1 | 80 | 22 | 20 | 24 | 21 |
| boundary_shift | 2 | 83 | 41 | 38 | 50 | 43 |
| boundary_shift | 3 | 82 | 49 | 46 | 56 | 50 |
| boundary_shift | 4 | 83 | 52 | 49 | 61 | 53 |
| thin_obstacle_miss | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 2 | 0 | 0 | 0 | 0 | 0 |
| thin_obstacle_miss | 4 | 5 | 5 | 5 | 6 | 6 |
| thin_obstacle_miss | 6 | 12 | 13 | 13 | 14 | 13 |
| boundary_noise | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary_noise | 1 | 49 | 12 | 11 | 20 | 11 |
| boundary_noise | 2 | 52 | 17 | 13 | 29 | 18 |
| boundary_noise | 3 | 57 | 24 | 22 | 35 | 25 |

## Stranding reasons (all planners, all levels)

- boundary_shift: endpoint 1135, no_path 230
- region_dropout: endpoint 3675, no_path 1270
- boundary_noise: endpoint 420, no_path 135

## Part B: per-frame rates, pooled over planners, raw → filtered

| bad frame model | p_bad | collision % | margin violation % | stranded % |
|---|---|---|---|---|
| boundary_shift 3 | 0 | 0 → 0 | 0 → 0 | 4 → 1 |
| boundary_shift 3 | 0.05 | 4 → 1 | 4 → 1 | 3 → 1 |
| boundary_shift 3 | 0.1 | 7 → 1 | 7 → 1 | 3 → 1 |
| boundary_shift 3 | 0.2 | 14 → 4 | 14 → 4 | 2 → 1 |
| boundary_shift 3 | 0.4 | 30 → 25 | 31 → 26 | 2 → 1 |
| region_dropout 0.4 | 0 | 0 → 0 | 0 → 0 | 4 → 2 |
| region_dropout 0.4 | 0.05 | 0 → 0 | 0 → 0 | 8 → 2 |
| region_dropout 0.4 | 0.1 | 0 → 0 | 0 → 0 | 14 → 4 |
| region_dropout 0.4 | 0.2 | 0 → 0 | 0 → 0 | 26 → 7 |
| region_dropout 0.4 | 0.4 | 0 → 0 | 0 → 0 | 35 → 14 |
| thin_obstacle_miss 4 | 0 | 0 → 0 | 0 → 0 | 2 → 1 |
| thin_obstacle_miss 4 | 0.05 | 0 → 0 | 0 → 0 | 3 → 1 |
| thin_obstacle_miss 4 | 0.1 | 1 → 0 | 2 → 0 | 3 → 2 |
| thin_obstacle_miss 4 | 0.2 | 1 → 0 | 3 → 1 | 3 → 1 |
| thin_obstacle_miss 4 | 0.4 | 3 → 3 | 7 → 6 | 2 → 0 |

## Part B: collision % per planner at p_bad = 0.2, raw → filtered

| bad frame model | A* | wA* λ16 | wA* λ64 | Vor nearest | Vor best64 |
|---|---|---|---|---|---|
| boundary_shift 3 | 15 → 4 | 13 → 4 | 13 → 4 | 15 → 4 | 14 → 4 |
| region_dropout 0.4 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| thin_obstacle_miss 4 | 1 → 0 | 1 → 0 | 2 → 0 | 2 → 1 | 1 → 0 |

## Part B: outlier-gate detection (all streams with p_bad > 0)

| bad frame model | bad frames | rejected | precision | recall |
|---|---|---|---|---|
| boundary_shift 3 | 153 | 0 | – | 0% |
| region_dropout 0.4 | 157 | 160 | 89% | 91% |
| thin_obstacle_miss 4 | 157 | 0 | – | 0% |
