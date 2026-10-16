# Q4 data: first-person video with ground-truth floor masks

Q4 runs SAM 2 on first-person video and turns each floor mask into a
bird's-eye-view (BEV) occupancy grid for the Q1/Q2 planners. Grading needs
the same pipeline on ground truth: GT floor mask → same BEV transform → GT
grid, then Q3's path-safety grading. That needs a dataset with:

- consecutive frames (video), from robot or phone height;
- per-frame semantic labels that identify the floor;
- intrinsics, and camera height and pitch over the floor (or depth + pose
  to fit them).

## Choice: TartanGround (indoor subset)

**TartanGround** (CMU AirLab + ETH, IROS 2025;
<https://huggingface.co/datasets/theairlabcmu/TartanGround>, CC BY 4.0, no
agreement to sign). It is photorealistic Unreal Engine data from simulated
ground robots: 63 environments, 878 trajectories at 10 Hz, each with six
640×640 90° stereo pinhole cameras, per-frame semantic labels, depth,
and 6-DoF poses. The paper samples robot height per trajectory in
0.5–1.5 m. The front cameras in this subset are fitted at 0.60 m
(robot-like) and 1.39 m (close to a hand-held phone).

### Candidates considered

| Dataset | Video | Floor GT | Height / pitch | License / access | Size / effort | Verdict |
|---|---|---|---|---|---|---|
| **TartanGround** | 10 Hz trajectories, 600–8000 frames | per-frame class ids with a per-environment `seg_label_map.json`; most indoor environments have a `floor` / `carpet` class, but some label floor and walls alike as `building` | exact poses (NED) + depth | CC BY 4.0, open on HF | one zip per trajectory, camera and modality (0.3–2.8 GB), but it supports HTTP range reads, so a subset is ~0.25 GB | **chosen** |
| TartanAir v1/v2 | yes | per-environment ids, same idea | poses + depth | CC BY 4.0 | large zips | camera flies freely (drone-like 6-DoF), not at ground height; TartanGround is the ground-robot version |
| ScanNet (v2 / ++) | real RGB-D video | NYU40/ScanNet labels, floor class | poses + depth | requires signing a terms-of-use form | tens of GB per scene set | real data, but needs a signed agreement; labels are mesh projections with ragged edges; handheld at chest height |
| NYU Depth v2 (labeled) | **no**: 1449 isolated labeled frames; the raw video has no labels | floor class | Kinect depth | non-commercial research | 2.8 GB | no labeled sequences |
| Replica / Habitat renders | renderable | semantic meshes | exact | Replica research license; Habitat install | needs habitat-sim (heavy, no Apple Silicon wheels for older versions) + 12 GB scenes | too much setup for a laptop |
| Hypersim | 100-frame random-walk trajectories, sparse spacing | NYU40 floor | depth + poses, but tilt-shift cameras with nonstandard per-scene projection | CC BY-SA 3.0 | 1–20 GB zips (range reads possible) | not video-like; eye-height artist cameras; odd intrinsics |
| SceneNet RGB-D | random trajectories | NYU13 | depth + poses | GPL-style research | 15 GB+ tarballs, no partial download | too big |
| SANPO | egocentric video | panoptic | depth + poses | CC BY 4.0 | large | outdoor sidewalks only |
| SUN RGB-D, ADE20K | no video | yes | partial | mixed | – | no sequences |

TartanGround wins on everything except realism. It is synthetic, so the
rendered textures are cleaner than a real phone camera's, and Q4's SAM 2
numbers should be read as an upper bound for real indoor video.
ScanNet is the natural real-data follow-up if the user signs its terms of
use.

### The subset

`docs/q4_manifest.json` pins 3 sequences × 150 consecutive frames (15 s at
10 Hz each) from the front-left camera (`lcam_front`):

| Sequence | Env / trajectory | Robot | Frames | Camera height | Notes |
|---|---|---|---|---|---|
| `Hospital_P1000` | Hospital, `Data_diff/P1000` | differential drive | 2150–2299 | 0.60 m | rooms, doors, chairs with thin legs |
| `AbandonedSchool_P1001` | AbandonedSchool, `Data_diff/P1001` | differential drive | 325–474 | 0.61 m | long corridors, debris, dirt decals |
| `Office_P0000` | Office, `Data_omni/P0000` | omnidirectional | 75–224 | 1.39 m | open lobby, desks, carpet |

Across all three sequences, 36–39% of image pixels are floor, and pitch and
roll stay below 0.5°. Between 20% and 49% of the recommended BEV grid
(below) comes out free.

Each frame has a 640×640 RGB PNG and a uint8 label PNG. A depth PNG comes
with every 10th frame (only used to fit the floor plane). Each sequence also
has `pose_lcam_front.txt` (the whole trajectory), `metadata.json`, and the
environment's `seg_label_map.json`. Total: 248 MB on disk (954 files). A fresh fetch transfers about the
same amount.

**How the windows were picked.** Some trajectories are not flat: Hospital
P1000 climbs 13 m over ramps and stairs, with pitch up to 34°. So the
fetcher's `SELECTION` uses, per trajectory, the 150-frame window (starting
on a multiple of 25) with the most motion (path length + 2 × total yaw
change in rad) among windows where camera z varies < 5 cm and |pitch| < 1°.
That keeps the flat-floor assumption true by construction. Stairs and ramps
are a known failure mode, documented below, not something this subset
tests.

**Why not Supermarket?** It was in the first selection. Its label map has
`floorrug` and `ground`, but the actual floor (like the walls and ceiling)
is labelled `building`, so no class name separates the floor from the
walls. Restaurant has the same problem. AbandonedSchool replaced it. Before
adding an environment, check that the bottom rows of its frames are mostly
a `FLOOR_NAMES` class, and that `load_sequence`'s floor fit gives a small
`floor_fit_spread_m`.

### Fetching

```bash
.venv/bin/python -I scripts/fetch_q4_data.py --dry-run   # list sequences and sizes
.venv/bin/python -I scripts/fetch_q4_data.py             # download + verify
```

The URLs are pinned to dataset revision `388faf9c`. The script never
downloads a whole zip: `RemoteFile` gives Python's `zipfile` a seekable file
backed by HTTP Range requests (each member's exact byte span in one request). It
reads the zip's central directory (~0.1–0.3 MB, in cached 1 MB blocks) and
then exactly the members listed in the manifest, in archive order. Every written file must match the manifest
sha256. Members map to fixed output paths, so an archive cannot write
outside `data/raw/q4/`. Files that already verify are not fetched again.
`--probe` prints each trajectory's frame count, and `--select`
(maintainer) re-fetches `SELECTION` and rewrites the manifest.

## Loading: `voronav.vision.dataset`

```python
from voronav.vision.dataset import list_sequences, load_sequence
for path in list_sequences("data/raw/q4"):
    seq = load_sequence(path)
    rgb = seq.frames[i]          # uint8 [640, 640, 3], read lazily
    gt = seq.floor_masks[i]      # bool [640, 640], True = floor
    cam = seq.camera(i)          # CameraPose(height_m, pitch_rad, roll_rad)
    seq.intrinsics               # fx = fy = 320, cx = cy = 319.5
```

- **Floor GT.** A pixel is floor when its label's class name is one of
  `floor, ground, carpet, rug, floorrug, floormat` or a flat floor decal
  (`floordirt`, painted court lines), matched exactly against the
  environment's label map. `stairs`, `staircase`, `floorstand` and
  `floorradiator` are not floor.
- **Intrinsics.** TartanGround cameras are 90° pinholes with square pixels,
  so f = W/2 = 320 px. The principal point is the image center, (W−1)/2 in
  the convention where pixel centers sit at integer coordinates. (TartanAir's
  own docs say 320; the half-pixel difference doesn't matter here.)
- **Pitch and roll** come from each frame's pose quaternion. The poses are
  NED world with the body frame x forward, y right, z down, so
  pitch = asin(R[2,0]) (nose down > 0) and roll = atan2(R[2,1], R[2,2]).
  In the chosen windows both are ~0, because the robots keep the camera
  level.
- **Camera height** = floor_z − camera_z (NED, z points down). floor_z is
  fitted once per sequence as the median world z of GT-floor pixels with
  depth < 10 m, back-projected from the depth frames. `floor_fit_spread_m`
  (the MAD) checks that the floor is flat: 0.1 mm (Hospital), 1.1 mm
(AbandonedSchool) and 0.2 mm (Office). Also checked against depth: for a
level camera, a floor pixel in row v should have depth h·f/(v−c). That
holds to within 2% for 99.3% / 99.8% / 100% of GT floor pixels. Do **not** use
  `robot_height` from `metadata.json` as the camera height. On
  Hospital_P1000 it says 1.66 m (1.33 m for AbandonedSchool, 1.28 m for
Office, against fitted 0.60 / 0.61 / 1.39 m), while both the floor fit and the
  bottom-row depth (a level 90° camera sees the floor at distance h on its
  bottom edge) give 0.60 m.
- **Depth decoding.** The depth PNGs are float32 values whose 4 bytes are
  stored in OpenCV's BGRA channel order (tartanairpy does
  `cv2.imread(...).view('<f4')`). Read with PIL as RGBA and taken
  literally, they decode to plausible-looking garbage (0.5–8 m, a 0.35 m
  camera). `read_depth` swaps the channels.

## BEV transform: `voronav.vision.bev`

The flat-ground homography maps a floor point (X forward, Y left, metres,
origin on the floor under the camera) to a pixel:

    [u, v, w]^T = K Rᵀ [e1 | e2 | −C] [X, Y, 1]^T,   C = (0, 0, h)

Here R is the camera's (right, down, forward) axes in the ground frame,
built from pitch and roll, and w is the depth along the optical axis
(w ≤ 0: behind the camera).

- `mask_to_bev(mask, intrinsics, cam, px_per_m, extent_m, max_range_m)`
  returns a bool grid (True = obstacle or unseen). The robot sits at the
  bottom centre with forward up, and `extent_m = (forward, width)`. Every
  cell is sampled at `supersample`² points (nearest pixel). A cell is free
  only if all of its samples are in front of the camera, inside the image,
  within `max_range_m`, and on floor pixels. Everything else is an
  obstacle. Q3 showed that leaking floor into walls causes collisions while
  shrinking the floor only strands the planner, so unseen space must never
  read as free.
- `bev_from_points(mask, ground_xy, image_uv, ...)` does the same from ≥ 4
  marked floor points (normalised DLT). Use it to calibrate a real phone
  with tape marks at measured spots instead of trusting intrinsics, height
  and pitch.
- `fov_mask(...)`: the visible wedge. `blind_range_m(...)`: how far ahead
  the floor is hidden below the image's bottom edge.

### Recommended planning parameters

**`px_per_m = 20` (5 cm cells), `extent_m = (5, 8)` → a 100 × 160 grid,
`max_range_m = 5`, `supersample = 2`** (these are `mask_to_bev`'s
defaults).

- 20 px/m matches the Q2 HouseExpo maps, so Q2's clearance settings carry
  over.
- Range is limited by how fast the homography's ground footprint grows.
  Along the view direction, one pixel covers about r²/(f·h) metres of floor
  (f = 320 px):

  | camera h | r = 2 m | 4 m | 5 m | 6 m | 8 m |
  |---|---|---|---|---|---|
  | 0.60 m (Hospital, AbandonedSchool) | 2.1 cm | 8.3 cm | 13 cm | 19 cm | 33 cm |
  | 1.39 m (Office) | 0.9 cm | 3.6 cm | 5.6 cm | 8.1 cm | 14 cm |

  At 5 m from the 0.6 m cameras, one pixel already spans ~2.6 cells in
  depth, so one mislabelled pixel blocks or frees a 13 cm streak. 5 m is
  the cap; Office alone could go to 6–7 m.
- 8 m wide covers the 90° wedge out to X = 4 m, which is all the width
  that matters before the range cap.

**The start cell is unseen.** With a level camera, the floor closer than
`blind_range_m` (= h for a 90° FOV: 0.60–1.39 m here) is below the image,
so the robot's own cell is an obstacle. The planning harness has to choose:
start at the first free cell straight ahead, or explicitly clear a
footprint disc. Either way it should be stated in the results; the BEV
doesn't hide it.

### Assumptions and failure modes

- **Flat floor.** Everything is projected onto Z = 0. A point at height z
  below the camera lands at h/(h−z) times its true distance: the 0.4 m
  edge of a shelf seen from a 0.6 m camera lands 3× too far, and anything
  above camera height lands above the horizon. That is only harmless
  because such points aren't floor in the mask, so they become obstacles
  at the wrong place, not free space. Ramps and stairs warp the grid: a 5°
  slope at 4 m is ~35 cm of height error. The subset avoids them on
  purpose.
- **Height and pitch errors** rescale the grid. A height error of Δh
  scales all distances by (h+Δh)/h. A pitch error of Δθ at range r moves
  points by about r²·Δθ/h, which gets large quickly: 1° at 4 m from 0.6 m
  is ~47 cm. On a real phone, prefer `bev_from_points` or an IMU pitch.
- **Far-range blur.** See the footprint table above. Nearest-pixel sampling
  plus the all-samples-free rule keeps it conservative: a cell is free only
  if every sample hits floor.
- **Occlusion.** Floor behind an obstacle is not visible, so it stays an
  obstacle (unseen). The BEV never guesses through objects. Thin
  overhangs (a desk edge above visible floor) are the exception: the floor
  under them is visible, so it is marked free even though the robot can't
  fit under them. The flat-floor BEV can't see this, and it is the same
  "thin obstacle miss" mode Q3 tested.
- **Synthetic labels.** The GT is perfect at object boundaries, which is
  stricter than any real annotation. SAM 2's errors against it are real
  errors, not label noise.

Attribution: TartanGround © Carnegie Mellon University AirLab, CC BY 4.0.
