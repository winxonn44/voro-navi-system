# Q3 segmentation-error models

Q3 asks how segmentation errors carry through to path safety. The method:
corrupt a true occupancy grid, plan on the corrupted grid, then grade the path
on the true grid. This file describes the corruptions
(`src/voronav/corruption.py`) and the bad-frame sequence and temporal filter
(`src/voronav/temporal.py`).

## Conventions

- Occupancy grids are `bool[H, W]`, with True meaning obstacle. The perception
  pipeline segments **floor**, so each model is written as an error in the
  floor mask.
- The two error directions are not symmetric:
  - **False floor** (an obstacle marked as floor) is the dangerous one,
    because the planner may route through something real.
  - **False obstacle** (floor marked as obstacle) fails safe, but it can
    disconnect the map and strand the robot.
- Every model has the signature `fn(occ, level, rng) -> new_occ`. It returns a
  new grid and never changes its input. `level = 0` returns an identical copy.
  Each model is reproducible for a given `rng` seed. Registry:
  `CORRUPTIONS[name]`.
- **The frame rule:** obstacle pixels on the outermost ring of the map are
  never freed. That ring stands for the known extent of the view, not for a
  segmentation output, so maps with a border frame stay bounded whatever the
  corruption.
- Distances are Euclidean and come from SciPy's exact distance transform.
  `signed_floor_distance(occ)` puts the floor/obstacle boundary halfway between
  pixel centres. It is positive on floor, so the pixels on either side of the
  boundary sit at ±0.5.

## Models

| name | `level` | direction | SAM failure it stands for |
|---|---|---|---|
| `boundary_shift` | signed px | +: floor grows (dangerous), −: floor shrinks (safe) | a systematic bias at mask edges: over- or under-segmenting the floor around every wall |
| `region_dropout` | fraction of floor, [0, 1] | safe (only removes floor) | clipping the target, or returning one part instead of the whole after a point-and-click |
| `thin_obstacle_miss` | width threshold, px | dangerous (only adds floor) | merging away fine detail such as chair legs, poles, railings and table legs |
| `boundary_noise` | amplitude, px | both | jagged, ragged mask edges from low-res mask decoding and upsampling |

### `boundary_shift(occ, level, rng)`
- **Level > 0:** an obstacle pixel becomes floor if the nearest floor pixel
  centre is within `level`.
- **Level < 0:** a floor pixel becomes obstacle if the nearest obstacle pixel
  centre is within `|level|`. This is a disc dilation of the obstacles.
- The model is deterministic (`rng` is unused) and monotonic in `level`. On an
  axis-aligned square it is exact: shrinking keeps sharp corners, growing gives
  rounded ones.
- **Limitations:**
  - Real edge bias varies with contrast and lighting; here it is uniform.
  - With `level > 0`, every obstacle thinner than about `2·level` disappears
    completely. That overlaps with `thin_obstacle_miss`, and it is realistic.

### `region_dropout(occ, level, rng, cell_px=None)`
- **How it works:** pick random floor seeds, split all pixels into Voronoi
  cells around them with the JFA engine (`voronav.voronoi.voronoi`, numpy
  backend), then mark whole cells as obstacle in random order. A cell is kept
  only if adding it moves the removed total closer to `level · n_free`.
- **`cell_px`:** the typical cell side. The default is
  `max(4, sqrt(n_free) / 6)`, about 36 cells per map, which keeps the result
  within a fraction of one cell of the target. The tests check it is within
  0.05.
- **Limitations:**
  - Cells are Euclidean, not geodesic, so one cell can take floor on both sides
    of a thin wall.
  - Dropped cells are scattered, not one contiguous "other half of the room".
    Contiguous growth over the cell adjacency graph would be closer to "part
    instead of whole" and is a possible extension.
  - Real clipping tends to follow image edges, such as rugs or a change in
    lighting, not random blobs.

### `thin_obstacle_miss(occ, level, rng)`
- **How it works:** label the obstacles into 8-connected components. A
  component's width is twice its largest distance-transform value, which is the
  diameter of its largest inscribed disc. A w-px bar measures w when w is even
  and w + 1 when w is odd. Components narrower than `level` are erased.
  Components that touch the map edge are always kept.
- The model is deterministic (`rng` is unused).
- **Limitations:**
  - It works per component. A thin railing joined to a thick wall survives,
    because the component as a whole is wide. A morphological opening would
    remove thin protrusions instead, at the cost of also nibbling corners.
  - Thin structures attached to the border frame are never erased. That
    includes interior walls joined to the outer wall. This is deliberate, so
    that wall gaps aren't invented, but it understates the danger on maps
    where walls are themselves only 1–2 px.
  - A miss is all-or-nothing per object. SAM sometimes keeps part of a leg.

### `boundary_noise(occ, level, rng, corr_px=None)`
- **How it works:** take white noise, smooth it with a Gaussian of width
  `corr_px` (default `max(2, 2·level)`), standardise it and clip it to
  [−1, 1]. A pixel is floor when
  `signed_floor_distance + level · field > 0`.
- Only pixels within `level` of the true boundary can flip, so the error stays
  inside that band. Errors go both ways and come in runs along the edge, not as
  salt-and-pepper.
- **Limitations:**
  - The noise is stationary, the same everywhere. Real errors cluster at
    low-contrast edges.
  - Noise with a zero mean leaves the floor area roughly unbiased. Combine it
    with `boundary_shift` for a biased segmenter.

## `mask_error_stats(true_occ, est_occ) -> dict`

| key | meaning |
|---|---|
| `free_iou` | IoU of the two floor masks (1.0 if neither has any floor) |
| `false_free_frac` | true obstacle pixels marked floor ÷ true obstacle pixels. **The dangerous error.** |
| `false_obstacle_frac` | true floor pixels marked obstacle ÷ true floor pixels |
| `boundary_error_px` | mean distance from each estimated boundary pixel to the nearest true boundary pixel. Boundary pixels are obstacle pixels with a 4-neighbour of floor; the map edge does not count. 0 if neither map has a boundary, nan if only one does |
| `boundary_error_max_px` | the same, maximum instead of mean (a one-sided Hausdorff distance) |
| `n_false_free`, `n_false_obstacle` | raw pixel counts |

`boundary_error_px` is one-sided. It measures spurious edges, such as the
inside edge of a dropped region, but not edges that are missing. A pole that
vanishes therefore scores 0 here, and that error shows up in
`false_free_frac` instead.

## Bad frames and the temporal filter

### `bad_frame_sequence(occ, n_frames, p_bad, rng, base=("boundary_noise", 1.0), bad=("region_dropout", 0.4), return_flags=False)`
- Every frame gets fresh noise. Each frame independently gets the `bad`
  corruption with probability `p_bad`, and the mild `base` corruption
  otherwise.
- `base` and `bad` are pairs of `(registry name, level)`, or `None` for no
  corruption. Examples: `("thin_obstacle_miss", 6.0)`, or any other registry
  name.
- With `return_flags=True` it returns `(frames, is_bad)` so the harness can
  tell which frames were bad.

### `temporal_filter(frames, window=5, outlier_iou=0.8, patience=3)`
The filter is causal: frame t only sees frames up to t. For each frame:
1. **Gate:** if the frame's floor IoU with the last accepted mask is below
   `outlier_iou`, reject it and keep using the last accepted mask. The first
   frame is always accepted.
2. **Vote:** output obstacle wherever at least half of the last `window`
   accepted masks say obstacle. Ties go to obstacle, the safe side.
3. **Reset:** after `patience` rejections in a row, accept the current frame
   and restart the history from it. Without this, a bad first frame would lock
   the filter forever. It also means a **sustained failure gets through and is
   not hidden**, which matches the intent: a few bad frames are tolerable, a
   persistent failure is not and should show up downstream.
   `patience=None` disables the reset.

What each stage catches:
- **The gate** catches large area errors, such as dropout. IoU barely moves
  when a pole is lost, so the gate does not catch thin-obstacle misses.
- **The vote** catches transient thin-obstacle misses: one frame without the
  pole is outvoted by the frames that still have it.
- **Neither** catches a miss that persists across most of the window.

Limitations:
- The scene is assumed static and the camera fixed. In Q4 the masks must first
  be registered into a common frame, such as the map frame from odometry,
  before IoU and voting mean anything.
- `outlier_iou` is an absolute threshold. On maps with a lot of boundary and
  little floor, `base` noise alone lowers frame-to-frame IoU, so the threshold
  may need to scale with how cluttered the map is.
- With fewer than `window` accepted frames, the tie-to-obstacle rule makes the
  early outputs conservative. With 2 frames, for example, the output is the
  union of their obstacles.
