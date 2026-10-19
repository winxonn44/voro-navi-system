# Q4 summary

3 sequences, 13860 frame rows, 165792 graded plans.

## Sequences

| sequence | frames | camera height m | path m | yaw change ° | GT free % of grid | frames with goals | goals |
|---|---|---|---|---|---|---|---|
| AbandonedSchool_P1001 | 150 | 0.60 | 27.2 | 202 | 31 | 144 | 432 |
| Hospital_P1000 | 150 | 0.60 | 15.7 | 440 | 21 | 126 | 374 |
| Office_P0000 | 150 | 1.39 | 13.0 | 185 | 40 | 150 | 450 |

## Mask quality per model (all sequences, erosion 0)

| model | rule | image IoU | image false floor % | image floor lost % | BEV IoU | BEV false free % (in view) | frames with leak ≥ 1.5 cells % |
|---|---|---|---|---|---|---|---|
| tiny | top_score | 0.964 | 0.01 | 3.6 | 0.916 | 0.09 | 5 |
| tiny | largest_consistent | 0.964 | 0.01 | 3.6 | 0.916 | 0.09 | 5 |
| small | top_score | 0.968 | 0.03 | 3.2 | 0.930 | 0.30 | 10 |
| small | largest_consistent | 0.969 | 0.04 | 3.0 | 0.933 | 0.31 | 10 |

## Mask quality per sequence and condition

| sequence | condition | image IoU | min image IoU | image false floor % | image floor lost % | BEV IoU | BEV false free % | leak depth p90 (cells) | boundary cells leaked % | small obstacles erased |
|---|---|---|---|---|---|---|---|---|---|---|
| AbandonedSchool_P1001 | tiny largest e0 | 0.969 | 0.630 | 0.01 | 3.1 | 0.913 | 0.13 | 0.2 | 1.1 | 12 |
| AbandonedSchool_P1001 | tiny largest e2 | 0.958 | 0.621 | 0.01 | 4.2 | 0.886 | 0.13 | 0.2 | 1.1 | 10 |
| AbandonedSchool_P1001 | tiny top e0 | 0.969 | 0.630 | 0.01 | 3.1 | 0.913 | 0.13 | 0.2 | 1.1 | 12 |
| AbandonedSchool_P1001 | tiny top e2 | 0.958 | 0.621 | 0.01 | 4.2 | 0.886 | 0.13 | 0.2 | 1.1 | 10 |
| AbandonedSchool_P1001 | small largest e0 | 0.978 | 0.726 | 0.02 | 2.2 | 0.935 | 0.17 | 0.3 | 1.3 | 10 |
| AbandonedSchool_P1001 | small largest e2 | 0.967 | 0.723 | 0.01 | 3.3 | 0.912 | 0.14 | 0.3 | 1.2 | 10 |
| AbandonedSchool_P1001 | small top e0 | 0.973 | 0.725 | 0.02 | 2.6 | 0.927 | 0.15 | 0.3 | 1.2 | 10 |
| AbandonedSchool_P1001 | small top e2 | 0.962 | 0.720 | 0.01 | 3.8 | 0.902 | 0.13 | 0.2 | 1.1 | 9 |
| Hospital_P1000 | tiny largest e0 | 0.949 | 0.000 | 0.00 | 5.0 | 0.842 | 0.00 | 0.0 | 0.0 | 2 |
| Hospital_P1000 | tiny largest e2 | 0.932 | 0.000 | 0.00 | 6.8 | 0.794 | 0.00 | 0.0 | 0.0 | 1 |
| Hospital_P1000 | tiny top e0 | 0.949 | 0.000 | 0.00 | 5.0 | 0.842 | 0.00 | 0.0 | 0.0 | 2 |
| Hospital_P1000 | tiny top e2 | 0.932 | 0.000 | 0.00 | 6.8 | 0.794 | 0.00 | 0.0 | 0.0 | 1 |
| Hospital_P1000 | small largest e0 | 0.979 | 0.907 | 0.07 | 2.0 | 0.896 | 0.71 | 0.6 | 2.4 | 2 |
| Hospital_P1000 | small largest e2 | 0.965 | 0.897 | 0.06 | 3.4 | 0.848 | 0.58 | 0.5 | 1.9 | 2 |
| Hospital_P1000 | small top e0 | 0.979 | 0.907 | 0.07 | 2.0 | 0.896 | 0.71 | 0.6 | 2.4 | 2 |
| Hospital_P1000 | small top e2 | 0.965 | 0.897 | 0.06 | 3.4 | 0.848 | 0.58 | 0.5 | 1.9 | 2 |
| Office_P0000 | tiny largest e0 | 0.972 | 0.901 | 0.01 | 2.8 | 0.981 | 0.11 | 0.2 | 0.4 | 2 |
| Office_P0000 | tiny largest e2 | 0.958 | 0.879 | 0.00 | 4.2 | 0.968 | 0.01 | 0.0 | 0.0 | 2 |
| Office_P0000 | tiny top e0 | 0.972 | 0.901 | 0.01 | 2.8 | 0.981 | 0.11 | 0.2 | 0.4 | 2 |
| Office_P0000 | tiny top e2 | 0.958 | 0.879 | 0.00 | 4.2 | 0.968 | 0.01 | 0.0 | 0.0 | 2 |
| Office_P0000 | small largest e0 | 0.952 | 0.861 | 0.02 | 4.7 | 0.963 | 0.10 | 0.1 | 0.4 | 2 |
| Office_P0000 | small largest e2 | 0.931 | 0.825 | 0.01 | 6.9 | 0.942 | 0.02 | 0.0 | 0.1 | 2 |
| Office_P0000 | small top e0 | 0.952 | 0.861 | 0.02 | 4.7 | 0.963 | 0.10 | 0.1 | 0.4 | 2 |
| Office_P0000 | small top e2 | 0.931 | 0.825 | 0.01 | 6.9 | 0.942 | 0.02 | 0.0 | 0.1 | 2 |

## SAM's multimask choice on the prompt frame

| sequence | model | rule | chosen | candidate scores | candidate areas (px) |
|---|---|---|---|---|---|
| AbandonedSchool_P1001 | tiny | top_score | 0 | 0.95, 0.34, 0.73 | 141824, 41629, 126055 |
| AbandonedSchool_P1001 | tiny | largest_consistent | 0 | 0.95, 0.34, 0.73 | 141824, 41629, 126055 |
| AbandonedSchool_P1001 | small | top_score | 1 | 0.13, 0.82, 0.79 | 6086, 142811, 145884 |
| AbandonedSchool_P1001 | small | largest_consistent | 2 | 0.13, 0.82, 0.79 | 6086, 142811, 145884 |
| Hospital_P1000 | tiny | top_score | 0 | 0.93, 0.40, 0.60 | 146293, 48586, 60550 |
| Hospital_P1000 | tiny | largest_consistent | 0 | 0.93, 0.40, 0.60 | 146293, 48586, 60550 |
| Hospital_P1000 | small | top_score | 2 | 0.32, 0.77, 0.88 | 30527, 138271, 159635 |
| Hospital_P1000 | small | largest_consistent | 2 | 0.32, 0.77, 0.88 | 30527, 138271, 159635 |
| Office_P0000 | tiny | top_score | 0 | 0.82, 0.21, 0.34 | 164504, 30214, 54890 |
| Office_P0000 | tiny | largest_consistent | 0 | 0.82, 0.21, 0.34 | 164504, 30214, 54890 |
| Office_P0000 | small | top_score | 2 | 0.06, 0.11, 0.55 | 3445, 144847, 163525 |
| Office_P0000 | small | largest_consistent | 2 | 0.06, 0.11, 0.55 | 3445, 144847, 163525 |

## Collision % (path enters a true obstacle or unseen cell)

Raw → each filtered stream (vote, IoU gate, vote, leak gate).

| condition | A* | wA* λ64 | Vor nearest | Vor best64 |
|---|---|---|---|---|
| GT (control) | 0.0 → 0.0 → 0.0 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.0 → 0.0 |
| GT + shift=3 | 1.2 → 0.6 → 0.2 | 1.0 → 0.1 → 0.2 | 1.4 → 0.2 → 0.2 | 1.0 → 0.0 → 0.1 |
| GT + thin miss=4 | 0.0 → 0.0 → 0.0 | 0.1 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.1 → 0.0 → 0.0 |
| tiny largest e0 | 0.8 → 1.1 → 0.7 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.0 → 0.0 |
| tiny largest e2 | 0.8 → 0.9 → 0.6 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.0 → 0.0 |
| tiny top e0 | 0.8 → 1.1 → 0.7 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.0 → 0.0 |
| tiny top e2 | 0.8 → 0.9 → 0.6 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.0 → 0.0 |
| small largest e0 | 0.7 → 0.7 → 0.7 | 0.1 → 0.0 → 0.0 | 0.2 → 0.0 → 0.0 | 0.1 → 0.0 → 0.0 |
| small largest e2 | 0.6 → 0.6 → 0.6 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 |
| small top e0 | 0.8 → 0.9 → 0.9 | 0.1 → 0.0 → 0.0 | 0.2 → 0.0 → 0.0 | 0.1 → 0.0 → 0.0 |
| small top e2 | 0.6 → 0.6 → 0.6 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 |

## Margin violation % (true clearance < robot radius)

Raw → each filtered stream (vote, IoU gate, vote, leak gate).

| condition | A* | wA* λ64 | Vor nearest | Vor best64 |
|---|---|---|---|---|
| GT (control) | 0.0 → 0.3 → 0.2 | 0.0 → 0.1 → 0.1 | 0.0 → 0.5 → 0.4 | 0.0 → 0.1 → 0.1 |
| GT + shift=3 | 2.0 → 1.1 → 0.6 | 1.1 → 0.3 → 0.2 | 1.8 → 0.5 → 0.2 | 1.1 → 0.3 → 0.2 |
| GT + thin miss=4 | 0.0 → 0.3 → 0.2 | 0.1 → 0.1 → 0.1 | 0.0 → 0.5 → 0.4 | 0.1 → 0.1 → 0.1 |
| tiny largest e0 | 1.1 → 1.6 → 1.2 | 0.0 → 0.2 → 0.2 | 0.0 → 0.2 → 0.2 | 0.0 → 0.2 → 0.2 |
| tiny largest e2 | 1.0 → 1.4 → 1.0 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 |
| tiny top e0 | 1.1 → 1.6 → 1.2 | 0.0 → 0.2 → 0.2 | 0.0 → 0.2 → 0.2 | 0.0 → 0.2 → 0.2 |
| tiny top e2 | 1.0 → 1.4 → 1.0 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 | 0.0 → 0.1 → 0.1 |
| small largest e0 | 0.9 → 1.0 → 1.0 | 0.1 → 0.0 → 0.0 | 0.2 → 0.2 → 0.2 | 0.1 → 0.0 → 0.0 |
| small largest e2 | 0.8 → 1.0 → 1.0 | 0.0 → 0.0 → 0.0 | 0.1 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 |
| small top e0 | 1.0 → 1.2 → 1.2 | 0.1 → 0.0 → 0.0 | 0.3 → 0.1 → 0.1 | 0.1 → 0.0 → 0.0 |
| small top e2 | 0.9 → 1.1 → 1.1 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 | 0.0 → 0.0 → 0.0 |

## Stranded % (all planners), raw → filtered

| condition | raw frames | vote, IoU gate | vote, leak gate | plans per stream |
|---|---|---|---|---|
| GT (control) | 0.0 | 6.2 | 5.7 | 5024 |
| GT + shift=3 | 0.0 | 6.1 | 7.6 | 5024 |
| GT + thin miss=4 | 0.0 | 6.2 | 5.7 | 5024 |
| tiny largest e0 | 12.7 | 16.6 | 17.1 | 5024 |
| tiny largest e2 | 19.7 | 22.9 | 23.6 | 5024 |
| tiny top e0 | 12.7 | 16.6 | 17.1 | 5024 |
| tiny top e2 | 19.7 | 22.9 | 23.6 | 5024 |
| small largest e0 | 15.8 | 19.9 | 19.8 | 5024 |
| small largest e2 | 25.8 | 26.8 | 26.7 | 5024 |
| small top e0 | 17.0 | 21.2 | 20.9 | 5024 |
| small top e2 | 26.9 | 28.0 | 28.1 | 5024 |

## Why plans strand (all planners)

*start unsafe*: every goal of the frame stranded at an endpoint for that planner, i.e. the start cell (the safe cell nearest the robot, just past the camera's blind zone) is not safe on the estimated grid. *goal unsafe*: another endpoint strand. *no path*: both endpoints safe, no route.

| condition | stream | start unsafe % of plans | goal unsafe % | no path % |
|---|---|---|---|---|
| GT (control) | vote, IoU gate | 0.8 | 5.2 | 0.2 |
| GT (control) | vote, leak gate | 0.6 | 4.9 | 0.2 |
| GT + shift=3 | vote, IoU gate | 1.5 | 4.5 | 0.1 |
| GT + shift=3 | vote, leak gate | 1.0 | 6.2 | 0.4 |
| GT + thin miss=4 | vote, IoU gate | 0.8 | 5.2 | 0.2 |
| GT + thin miss=4 | vote, leak gate | 0.6 | 4.9 | 0.2 |
| tiny largest e0 | raw frames | 4.7 | 7.6 | 0.4 |
| tiny largest e0 | vote, IoU gate | 4.2 | 12.2 | 0.2 |
| tiny largest e0 | vote, leak gate | 4.9 | 12.0 | 0.2 |
| tiny largest e2 | raw frames | 7.8 | 11.6 | 0.2 |
| tiny largest e2 | vote, IoU gate | 6.6 | 16.0 | 0.3 |
| tiny largest e2 | vote, leak gate | 7.8 | 15.5 | 0.3 |
| tiny top e0 | raw frames | 4.7 | 7.6 | 0.4 |
| tiny top e0 | vote, IoU gate | 4.2 | 12.2 | 0.2 |
| tiny top e0 | vote, leak gate | 4.9 | 12.0 | 0.2 |
| tiny top e2 | raw frames | 7.8 | 11.6 | 0.2 |
| tiny top e2 | vote, IoU gate | 6.6 | 16.0 | 0.3 |
| tiny top e2 | vote, leak gate | 7.8 | 15.5 | 0.3 |
| small largest e0 | raw frames | 9.5 | 6.1 | 0.3 |
| small largest e0 | vote, IoU gate | 8.8 | 10.7 | 0.5 |
| small largest e0 | vote, leak gate | 9.0 | 10.3 | 0.6 |
| small largest e2 | raw frames | 13.8 | 11.3 | 0.7 |
| small largest e2 | vote, IoU gate | 10.9 | 15.4 | 0.6 |
| small largest e2 | vote, leak gate | 10.9 | 15.2 | 0.6 |
| small top e0 | raw frames | 10.2 | 6.4 | 0.4 |
| small top e0 | vote, IoU gate | 9.5 | 11.2 | 0.5 |
| small top e0 | vote, leak gate | 9.5 | 10.9 | 0.6 |
| small top e2 | raw frames | 15.0 | 11.2 | 0.7 |
| small top e2 | vote, IoU gate | 11.6 | 15.8 | 0.6 |
| small top e2 | vote, leak gate | 12.1 | 15.4 | 0.6 |

## Gate detection on injected bad frames (GT stream + Q3 corruption on random frames)

| injected | stream | bad frames | rejected | precision | recall |
|---|---|---|---|---|---|
| GT + shift=3 | vote, IoU gate | 86 | 9 | 78 | 8 |
| GT + shift=3 | vote, leak gate | 86 | 64 | 88 | 65 |
| GT + thin miss=4 | vote, IoU gate | 86 | 0 | – | 0 |
| GT + thin miss=4 | vote, leak gate | 86 | 8 | 50 | 5 |

## Gates on SAM frames

A raw frame is *unsafe* if any planner collided on it. Rejected = the gate kept it out of the vote.

| condition | stream | frames | rejected % | unsafe raw frames | of those rejected |
|---|---|---|---|---|---|
| tiny largest e0 | vote, IoU gate | 420 | 1 | 8 | 0 |
| tiny largest e0 | vote, leak gate | 420 | 4 | 8 | 2 |
| tiny largest e2 | vote, IoU gate | 420 | 1 | 8 | 0 |
| tiny largest e2 | vote, leak gate | 420 | 5 | 8 | 2 |
| tiny top e0 | vote, IoU gate | 420 | 1 | 8 | 0 |
| tiny top e0 | vote, leak gate | 420 | 4 | 8 | 2 |
| tiny top e2 | vote, IoU gate | 420 | 1 | 8 | 0 |
| tiny top e2 | vote, leak gate | 420 | 5 | 8 | 2 |
| small largest e0 | vote, IoU gate | 420 | 1 | 8 | 0 |
| small largest e0 | vote, leak gate | 420 | 3 | 8 | 0 |
| small largest e2 | vote, IoU gate | 420 | 1 | 6 | 0 |
| small largest e2 | vote, leak gate | 420 | 5 | 6 | 0 |
| small top e0 | vote, IoU gate | 420 | 1 | 9 | 0 |
| small top e0 | vote, leak gate | 420 | 3 | 9 | 0 |
| small top e2 | vote, IoU gate | 420 | 1 | 6 | 0 |
| small top e2 | vote, leak gate | 420 | 5 | 6 | 0 |

## End-to-end latency per frame (ms, M4 MacBook Air, erosion 0, frame 0 excluded)

| model | SAM 2 median | SAM 2 p90 | bird's-eye | roadmap build | plan A* | plan wA* λ64 | plan Vor nearest | plan Vor best64 | end-to-end median | end-to-end p90 | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 523 | 603 | 1.3 | 2.7 | 0.0 | 0.3 | 0.1 | 0.3 | 527 | 608 | 1.90 |
| small | 633 | 714 | 1.3 | 2.7 | 0.0 | 0.3 | 0.1 | 0.3 | 637 | 718 | 1.57 |

End-to-end = SAM 2 + bird's-eye + roadmap build + one Voronoi best64 plan. The temporal filter adds 1.8 ms median.
