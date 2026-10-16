# Q3 harness: how does segmentation error carry through to path safety?

Q2 showed that the Voronoi roadmap matches tuned clearance-weighted A* on
clearance. Q3 asks what that clearance is worth when the map itself is wrong.
The method: **corrupt the true map, plan on the corrupted map, grade the path
on the true map.** The error models (`voronav.corruption`,
`voronav.temporal`) and their rationale are documented in
`docs/Q3_ERRORS.md`. This page covers the harness.

```bash
python scripts/bench_q3.py --config configs/bench_q3_quick.yaml   # synthetic only, ~30 s
python scripts/plot_q3.py runs/bench_q3_quick/results.jsonl --out runs/bench_q3_quick/figs
python scripts/bench_q3.py --config configs/bench_q3.yaml         # the study, run plugged in
```

## Part A: static errors

For every map, every corruption kind, every level and every draw
(`q3.corruptions`, `q3.draws`):

1. `est = CORRUPTIONS[kind](true_occ, level, rng)`. The rng is seeded from
   (seed, map index, kind, level, draw), so the draws do not depend on the order
   of the config.
2. Build the roadmap and the safe mask on `est` (the `roadmap:` block, as in Q2,
   including `bridge_lam: 64`).
3. Run every planner on the same seeded queries: grid A*, weighted A* for
   each λ in `q3.lambdas`, and Voronoi for each `q3.voronoi_variants` entry.
   RRT* is not included.
4. Grade with `path_metrics(path, exact_clearance(true_occ), true_safe)`,
   where `true_safe = free & exact clearance >= robot radius`.

Queries are sampled once per map in the **true** safe space with
`bench_plan.sample_queries`. Start and goal are in the same search region
(the search's own move rule, `grid_search.REACHABILITY`, which is what
`reachable_components` computes for plain grid search) and at least
`min_dist_frac` of the diagonal apart. Level 0 is the control, the true map:
there, every planner succeeds and stays safe (checked by the smoke test).

Outcomes per query row:

| field | meaning |
|---|---|
| `stranded` | an endpoint is not safe in the corrupted map (`strand_reason: endpoint`), or no path was found there (`no_path`, with the planner's `fail`) |
| `collision` | the path enters a true obstacle pixel (true min clearance 0 on the 0.25-px resampling) |
| `margin_violation` | true clearance < robot radius somewhere. Every collision is also a violation |
| `min_clearance_px`, `mean_clearance_px`, `length_px`, ... | `path_metrics` on the true map. `in_true_safe` is its `collision_free` |
| `free_iou`, `false_free_frac`, `false_obstacle_frac`, `boundary_error_px` | `mask_error_stats(true, est)` for that corrupted map |

**Map groups.** Every row has `group`: `cluttered` (cluttered_room) or
`structural` (every other map). They are analyzed and plotted apart. On
cluttered_room, any error that grows the floor also erases its 1–2 px chair
legs: `boundary_shift +1` grows the floor 1 px on each side of a 2 px leg, so
the leg is gone. Pooled, "boundary shift" there is partly "thin obstacles
missed". In the first quick run, all of the +1 px collisions were on
cluttered_room, and none on the structural maps.

A stranded row has `collision = margin_violation = False` and null metrics.
Rates are reported over all queries, so a stranded query counts as safe. That
is deliberate: stranding is the safe way to fail, and it gets its own chart.

## Part B: bad frames

`q3.frames` lists its own maps (Part B is skipped when that list is empty). For
each map, each `bad` model and each `p_bad`:

1. `raw, is_bad = bad_frame_sequence(occ, n_frames, p_bad, rng, base, bad, return_flags=True)`.
   Every frame gets the mild `base` error, and with probability p_bad it gets
   the `bad` one instead.
2. `filtered = temporal_filter(raw, window, outlier_iou, patience)`.
3. For each stream (raw, filtered), each frame and each query: build the
   roadmap on that frame, plan, and grade on the true map. Identical frames
   (common after the filter) are built only once.

**Collision is the primary metric** here; margin violation is secondary.

**Base calibration.** The base error must be harmless on its own, or the
filter has nothing to rescue. With `boundary_noise 1` as the base, 5–15 % of
plans already collided at p_bad = 0, and 10–34 % violated the margin
(calibration probe: 6 maps × 8 queries × 8 frames, p_bad = 0). At
`boundary_noise 0.5` the probe gave 0 % collisions and 0 % margin violations
in both map groups, and 1–3 % stranded, so the base still perturbs the map.
Both bench configs use 0.5. At amplitude 0.25 nothing changes at all.

**Filter diagnosis.** `temporal_filter` returns only masks, so `bench_q3.py`
carries `filter_trace`, a step-for-step copy that also reports the gate's
decision on each frame: `first`, `accept`, `reject` (an outlier; the last
accepted mask is kept) or `reset` (`patience` rejections in a row, so the
frame is accepted as a new start). It also reports the IoU the gate saw and
the frames in the vote. The harness checks every sequence: the copy's masks
must equal the real filter's, or the run aborts.
`tests/test_bench_smoke.py` checks the same on adversarial sequences. Rows:

* `frame`: per (sequence, stream, frame, query, planner), the outcome plus
  `frame_bad` (the raw frame's flag), `gate`, `gate_iou`, `n_vote` and
  `n_vote_bad` (injected-bad frames in the vote).
* `sequence`: per (stream, query, planner), the fractions of frames that
  collided, violated the margin, or stranded.
* `filter`: per sequence, the trace plus detection counts. Precision =
  rejected frames that were injected bad. Recall = injected bad frames that
  were rejected. A frame let through by the reset counts as not rejected.

## Maps and configs

| config | maps | load |
|---|---|---|
| `debug_q3.yaml` | 64² random, 96² cluttered_room, tiny Part B | ~1 s, used by the smoke test |
| `bench_q3_quick.yaml` | 2× random_obstacles 256², two_rooms 120, 2× cluttered_room 256²; Part B on cluttered_room 128 + random_obstacles 128, 3 queries, 12 frames, 3 bad models | 28 s measured on the M4 Air |
| `bench_q3.yaml` | 3× random 256², two_rooms 240, 3× cluttered_room 256², 6 HouseExpo, 2 each of MovingAI street / dao / bg512; Part B on 10 maps (3 cluttered_room, 2 random, two_rooms 120, 4 HouseExpo) × 5 queries × 20 frames, 3 bad models; `cooldown_s: 2` | not run yet. Scaling the quick run (~16 ms per frame at 128 px, 3 queries) gives about 4 min for Part A and 11 for Part B, about 15 min in total, inside the 20 min target |

Levels in both bench configs: `boundary_shift` [-3, -2, -1, 0, 1, 2, 3, 4] px,
`region_dropout` [0, 0.1, 0.2, 0.3, 0.5], `thin_obstacle_miss` [0, 2, 4, 6] px,
`boundary_noise` [0, 1, 2, 3] px. The configs are strict
(`voronav.config.Q3Config`): an unknown key, an unknown corruption, a planner
outside {voronoi, astar, weighted_astar}, or a malformed `[name, level]` pair
fails at load time.

**`cluttered_room`** (new, `voronav.map_sources`) is a 2 × 2 grid of rooms with
walls at least 6 px thick and a door in each inner wall. It is cluttered with
obstacles 1–4 px thin: table and chair legs (1–2 px dots), poles (2–4 px
squares), and railings (1–2 px lines, some with posts). It exists so that
`thin_obstacle_miss` has something to erase. The walls survive that
corruption (a 5 × 5 opening deletes about 110 thin components per map and
leaves the walls; see `tests/test_map_sources.py`). The clutter may stand
against walls, because real furniture does. An earlier version kept it 6 px
off the walls, which let wall-hugging A* avoid erased legs by construction
and biased the thin-obstacle comparison against Voronoi.

## Charts (`scripts/plot_q3.py`)

Part A (`q3_collision.png`, `q3_margin.png`, `q3_stranded.png`,
`q3_min_clearance.png`): one row per corruption kind, one column per map
group, one line per planner, ±1 binomial SE. Min clearance is over the paths
that were found only, so each point averages a different subset.
`q3_frames.png` shows collision, margin-violation and stranded frame rates
against p_bad, raw (solid) vs filtered (dashed), one column per bad model.
`q3_filter_detection.png` shows the gate's precision / recall per bad model,
and the IoU it saw on good vs injected-bad frames next to its threshold.
Palette and style match `plot_q2.py`.

## First numbers (quick config only, small n: a smoke reading, not the study)

Part A: 5 maps × 10 queries per level and planner (structural: 2 ×
random_obstacles 256 + two_rooms 120, 30 queries; cluttered: 2 × cluttered_room
256, 20 queries). Collision % for A* / weighted λ16 / weighted λ64 / Voronoi
nearest / Voronoi best64:

| corruption, level | structural | cluttered_room |
|---|---|---|
| control (level 0) | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| shift −1 … −3, dropout 0.1 … 0.5 | 0 everywhere (they strand) | 0 everywhere (they strand) |
| shift +1 | 0 / 0 / 0 / 0 / 0 | 45 / 55 / 40 / 60 / 60 |
| shift +2 (= robot radius) | **77 / 23 / 23 / 43 / 27** | 95 / 75 / 65 / 90 / 65 |
| shift +3 | 87 / 37 / 30 / 47 / 33 | 95 / 95 / 100 / 85 / 100 |
| thin miss 4 | 0 / 0 / 0 / 0 / 0 | **50 / 50 / 45 / 65 / 60** |
| thin miss 6 | 3 / 3 / 3 / 3 / 3 | 60 / 80 / 65 / 85 / 65 |
| noise 2 | 37 / 13 / 13 / 23 / 13 | 40 / 25 / 25 / 65 / 25 |
| noise 3 | 43 / 20 / 20 / 30 / 20 | 55 / 60 / 60 / 45 / 35 |

* **The fail-safe errors fail safe.** Shrinking the floor and dropping regions
  give 0 collisions at every level and in both groups. They strand instead.
  Shift −1 / −2 / −3 strands 17 / 23 / 30 % of structural queries. Dropout
  0.1 → 0.5 strands 33 → 100 %. Shift strandings are all unsafe endpoints. For
  dropout, 465 of 585 (79 %) are endpoints and the rest are searches with no
  path.
* **Structural maps: clearance pays.** Below the robot radius (shift +1, noise
  1), almost nothing collides (A* 3 % at noise 1). At shift +2, plain A* collides on 77 %. Weighted A* and
  Voronoi best64 collide on 23–27 %, about a third of that. The original
  Voronoi `nearest` sits in between at 43 %, because its legs onto and off the
  roadmap are shortest paths that hug the walls. Noise 2 shows the same order:
  37 vs 13 vs 23 %.
* **Clutter: clearance does not pay.** Missed thin obstacles cause 45–65 %
  collisions for every planner, and Voronoi is no better (60–65 % vs A* 50 %).
  It routes through mid-room, where the erased clutter stands. Clearance from
  the walls says nothing about an obstacle the map no longer has. On clutter
  maps, shift +1 already behaves like thin-obstacle miss (40–60 %) for the
  same reason.

Part B: 2 maps (cluttered_room 128, random_obstacles 128) × 3 queries × 12
frames, base `boundary_noise 0.5`. Collision % of frames, pooled over
planners, raw → filtered:

| bad model | p = 0 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| boundary_shift 3 | 0 → 0 | 0 → 0 | 1.7 → 0 | 5.8 → 0 | 10.8 → 11.7 |
| region_dropout 0.4, **stranded %** (never collides) | 0 → 0 | 13.9 → 2.8 | 9.7 → 0 | 29.2 → 11.1 | 37.5 → 30.6 |
| thin_obstacle_miss 4 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 (margin 7.5 → 5.0) | 0 → 0 (margin 10.0 → 7.5) |

(thin_obstacle_miss caused no Part B collisions with only 6 queries on 128 px maps; the study's 10
maps × 5 queries at 256 px is needed for that cell.)

**Detection by the outlier gate** (pooled over both maps and every p_bad):

| bad model | precision | recall | median gate IoU, bad vs good frames |
|---|---|---|---|
| region_dropout 0.4 | 18/23 = 78 % | 18/22 = 82 % | 0.59 vs 0.98 |
| boundary_shift 3 | 0/0 (nothing rejected) | 0/13 = 0 % | 0.88 vs 0.98 |
| thin_obstacle_miss 4 | 0/0 (nothing rejected) | 0/18 = 0 % | **0.99 vs 0.98** |

The hypothesis holds. The IoU gate cannot see a missed thin obstacle: erasing
legs and poles *adds* a sliver of floor, so a bad frame's IoU with the last
accepted mask is if anything higher than a good frame's. It cannot see a 3 px
shift either at this threshold: an IoU of 0.88 is above 0.8, and a tighter
threshold would depend on how much boundary the map has per unit of floor.
It catches only region dropout, which removes a large share of the floor. All 5
rejections that hit good frames came right after a bad dropout frame had been
accepted, usually as the first frame (good frames then look like outliers
against it). The patience reset fired 5 times: 3 times it recovered onto a
good frame, and 2 times it let a bad one in.

So, against shift and thin-miss frames, all the help comes from the majority
vote. It works while bad frames are a minority of the window: shift-3
collisions drop 5.8 → 0 % at p = 0.2, and thin-miss margin violations drop
about a quarter. It stops working at p = 0.4.

**Why the filter can be worse than raw (Grid A*, p_bad = 0.4, shift 3).**
Every filtered collision there (42/42) happened while injected-bad masks held
the majority of the vote. In the cluttered sequence, frames 0–2 were bad. The
first frame is always accepted, and the gate (IoU 0.88 > 0.8) let the next two
in. The vote then kept the bad map for 7 frames, against 6 bad frames in the
raw stream: a burst at the start lasts longer than the burst itself, because
the window has to fill with good frames before the vote turns. For bad
frames that add floor, a tie keeps obstacle (ties go to obstacle), so the bad
mask needs a strict majority. A burst of 3 in a window of up to 5 gives it
one. Raw and filtered Grid A* both ended at 29 % here. The earlier 94 % filtered vs
80 % raw was measured with base noise 1 on a single map. It was not
re-diagnosed, but this is the mechanism the trace now shows.

Reading so far: on structural maps the GVD's clearance pays off against
boundary errors when its routing is clearance-weighted (best64 ≈ weighted
A*, about a third of A*'s collisions at a 2 px shift). It does not pay off
against missed thin obstacles. The IoU-gated temporal filter does not detect
the dangerous errors (thin-miss, shift); only its majority vote helps, and
only while bad frames are rare.
