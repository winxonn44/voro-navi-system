# Q4 floor segmentation with SAM 2

Q4 runs real segmentation on first-person video. The pipeline is: floor mask,
then bird's-eye transform, then roadmap, then path. This file covers the first
step: how SAM 2 is installed and run offline on the M4 MacBook Air, the
interface in `src/voronav/vision/segment.py`, and how fast it is.

## Install (Python 3.14, torch 2.14, MPS)

Meta's package is called `SAM-2`. It is not published on PyPI under Meta's
name: the `sam2` project on PyPI is someone else's upload. So it is pinned to a
commit of [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
(`2b90b9f`, the repository's HEAD as of 2026-10-16; the repo has no tags).

```bash
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

```bash
.venv/bin/python -m pip install -e ".[vision]"
```

```bash
.venv/bin/python scripts/fetch_sam2.py
```

- `SAM2_BUILD_CUDA=0` skips SAM 2's only compiled part, a CUDA
  connected-components kernel. This is the documented CPU path (see the
  repository's INSTALL.md). Without it, SAM 2 skips one post-processing step,
  filling holes of up to 8 px in the low-resolution mask, and warns that it
  did. `Sam2Floor` silences that warning, along with a known-harmless
  `torch.jit` FutureWarning that SAM 2 triggers on Python 3.14.
- `--no-build-isolation` builds against the venv's torch. Without it, pip would
  download a second copy of torch into a temporary build environment.
- The second step (the `vision` extra in `pyproject.toml`) adds `torchvision`
  (0.29.1 matched torch 2.14.1) and SAM 2's own pure-Python dependencies:
  `hydra-core`, `omegaconf`, `iopath` and `tqdm`. All of them installed cleanly
  on Python 3.14.
- No part of the pipeline uses the network after this. Configs ship inside
  the package and weights load from `data/raw/sam2/`.

## Weights and licenses

`scripts/fetch_sam2.py` downloads the SAM 2.1 checkpoints into
`data/raw/sam2/`, which is gitignored. The URLs are the ones in the repository's
`checkpoints/download_ckpts.sh`.

| model | file | size | sha256 (pinned on first download, 2026-10-16) |
|---|---|---|---|
| tiny | `sam2.1_hiera_tiny.pt` | 156.0 MB | `7402e0d8…34be69` |
| small | `sam2.1_hiera_small.pt` | 184.4 MB | `6d1aa6f3…5dc4d38` |

Source: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/<file>`.
Meta publishes no checksums, so the hashes in the script were recorded on our
first download. They catch corruption or a silent upstream swap. They do not
prove the files are authentic.

**License.** The SAM 2 code and model checkpoints are both licensed under
**Apache 2.0** (the repository README's License section). The optional CUDA
kernel we skip is adapted from `cc_torch`, which has its own license
(`LICENSE_cctorch`). The demo's fonts are under the SIL OFL, but we use neither
the demo nor the kernel.

## Interface (`voronav.vision.segment`)

Conventions are the ones in `voronav.vision`: frames are uint8 `[H, W, 3]`
RGB, and masks are bool `[H, W]` with True = floor.

- **`FloorSegmenter`** is a protocol (runtime-checkable) with two methods.
  `segment_video(frames, prompts)` returns one mask per frame, and
  `segment_image(frame, prompts)` returns one mask. Any backend that has these
  two methods fits, including the fake segmenter in the tests.
- **`Prompts(positive, negative=(), box=None, frames=(0,))`** holds the prompts.
  Points are `(x, y)` pixels. Positives are on the floor and negatives are on
  walls or obstacles. `box` is `(x0, y0, x1, y1)`. `frames` lists the frames
  the prompts are applied on. Prompts are frozen and hashable, and
  `to_json()` / `from_json()` round-trip them.
- **`default_floor_prompts(h, w, frames=(0,))`** is a heuristic for a
  forward-facing camera. It puts three positives in the bottom band (50% width
  at 92% height, and 30% / 70% width at 85% height) and two negatives in the
  upper corners (8% in from each side, 8% down). It is a guess about framing.
  It fails when the camera pitches down at an obstacle, or when something sits
  right in front of the camera. On SAM 2's own bedroom sample clip it picks the
  bed, because the bed is what fills the bottom of the frame.
- **`Sam2Floor(model="tiny"|"small", device=None|"mps"|"cpu",
  select="largest_consistent"|"top_score", erode_px=0,
  precision=None|"fp32"|"bf16"|"fp16")`** wraps SAM 2.
  `device=None` means MPS when it is available. `precision=None` means bf16
  autocast on MPS and fp32 on CPU; half precision is refused on CPU, where it
  is about 90× slower (see Latency). Use `cache_tag` (for example
  `sam2.1-tiny/largest_consistent/erode0`) to identify the configuration.
  After each call, `last_choices` holds the candidate scores, the candidate
  areas and the chosen index for each prompt frame.
- **`select_mask(masks, scores, prompts, rule)`**, **`erode(mask, px)`** and
  **`CachedSegmenter(inner, cache_dir)`** are described below.

### How `Sam2Floor.segment_video` works

1. Frames are fed to SAM 2's video predictor from arrays. Its `init_state`
   only takes a JPEG folder or an MP4, so its frame loader is swapped, for that
   one call, for a loader that does the same preprocessing: PIL resize to
   1024², divide by 255, then ImageNet mean/std. The rest of `init_state` is
   SAM 2's own code.
2. On each prompt frame, SAM 2's image predictor runs on the same network
   with `multimask_output=True`. That gives 3 candidate masks with predicted
   IoUs. `select_mask` keeps one of them. The video predictor hides these
   candidates: with more than one click it returns a single mask.
3. The chosen mask goes into the video predictor with `add_new_mask`. Then
   `propagate_in_video` tracks it forward, and also backward if the first
   prompt frame is not frame 0, using SAM 2's memory.
4. Each output mask is thresholded at logit > 0 at the input resolution, then
   eroded by `erode_px`.

Both selection rules go through the same path, so the comparison isolates the
rule. The cost is one extra image-encoder pass per prompt frame.

### Multimask selection rule

The point-and-click failure this targets is a click that returns a *part* of
the floor (a rug, a lit patch, one tile region) instead of the whole floor.

- `top_score`: SAM's own choice, the candidate with the highest predicted IoU.
- `largest_consistent`: a candidate is *consistent* if it contains every
  positive point and no negative point. Among consistent candidates, the one
  with the most pixels wins. Equal areas go to the higher predicted IoU. If
  no candidate is consistent, the one with the fewest prompt violations wins,
  then the larger area, then the higher IoU.

Preferring the largest mask pushes toward the error Q3 found dangerous: a floor
mask that leaks into walls. The negatives, and `erode_px`, are the guard
against it. Q4 should report both rules, and also how often they disagree
(`last_choices` records it).

### Erosion

`erode(mask, px)` erodes the mask with a disk of radius `px`. Pixels outside
the image count as floor (`border_value=True`), so floor that runs off the
bottom of the frame stays attached to the edge. Only real boundaries move
inward, and slivers thinner than about 2·px disappear. This follows Q3: a
shrunken mask fails safe, and a leaking mask causes collisions.

### Cache

`CachedSegmenter(inner, cache_dir="runs/q4_masks")` wraps any segmenter.
`runs/` is gitignored. Its `segment_video(frames, prompts, sequence="name")`
stores the masks as a compressed `.npz` of bit-packed masks. The cache key is
built from the sequence name, the inner segmenter's `cache_tag` (model, rule,
erosion, precision), the prompts' JSON, and the frame count and size. Without
a sequence name, the frames' bytes are hashed instead. The device is
deliberately not part of the key: MPS and CPU masks differ by a few boundary
pixels at most. `hits` and `misses` count cache use.

## Latency (`scripts/time_sam2.py`)

**Frames.** The first full sweep ran before the Q4 dataset landed, so it used
the first 50 frames of SAM 2's own sample clip (`notebooks/videos/bedroom` in
the sam2 repository, 960×540). Once TartanGround arrived (see
`docs/Q4_DATA.md`), the default configurations were re-run on the first 50
frames of `Hospital_P1000` (640×640). SAM resizes every frame to 1024² before
encoding, so input resolution barely matters, and the two sets of runs agree
within 10 ms. To re-run:

```bash
.venv/bin/python scripts/time_sam2.py --frames data/raw/q4/Hospital_P1000/image -n 50
```

**Setup.** M4 MacBook Air (fanless), torch 2.14.1, one short run per
configuration, run back to back. The prompts are the default floor prompts on
frame 0, with `largest_consistent` selection. Raw per-frame series are in
`runs/q4_sam_timing_{fp32,mps_half,hospital_mps,hospital_cpu}.json`
(`runs/` is gitignored).

On TartanGround `Hospital_P1000`, 640×640, the dataset's resolution:

| model | device | precision | setup s | first frame s | frame median ms | frame p90 ms | fps |
|---|---|---|---|---|---|---|---|
| **tiny** | **mps** | **bf16 (default)** | 0.70 | 1.55 | **515** | **520** | **1.9** |
| **small** | **mps** | **bf16 (default)** | 0.44 | 1.52 | **549** | **553** | **1.8** |
| tiny | cpu | fp32 (default) | 0.79 | 1.72 | 1204 | 1249 | 0.8 |

With the default prompts, both models found a floor region covering 29–39% of
each frame, stable across the 50 frames. Tiny's top-score candidate was also
the largest consistent one. Small's top score (0.88) went to its largest
candidate. Grading that against the ground-truth masks is the Q4 harness's
job.

On the bedroom clip, 960×540, the full sweep:

| model | device | precision | setup s | first frame s | frame median ms | frame p90 ms | fps | frame 1 ms |
|---|---|---|---|---|---|---|---|---|
| tiny | mps | fp32 | 0.91 | 1.75 | 1228 | 1255 | 0.8 | 526 |
| small | mps | fp32 | 0.37 | 1.16 | 1164 | 1179 | 0.9 | 553 |
| tiny | cpu | fp32 | 0.25 | 2.13 | 1170 | 1238 | 0.9 | 812 |
| small | cpu | fp32 | 0.21 | 2.06 | 1272 | 1319 | 0.8 | 919 |
| **tiny** | **mps** | **bf16 (default)** | 0.87 | 1.30 | **515** | **522** | **1.9** | 331 |
| tiny | mps | fp16 | 0.38 | 1.60 | 530 | 542 | 1.9 | 318 |
| **small** | **mps** | **bf16 (default)** | 0.44 | 1.67 | **557** | **567** | **1.8** | 371 |
| small | mps | fp16 | 0.40 | 1.58 | 596 | 609 | 1.7 | 364 |

How to read the columns:

- *setup* is the model build plus the checkpoint load. Only the first
  configuration's setup includes reading the weights from disk; after that
  they come from the OS page cache.
- *first frame* is the warmup cost: frame loading and encoding, the multimask
  decode and selection, and `add_new_mask`.
- *frame* covers every propagated frame after the prompt frame.
- *frame 1* shows the ramp-up. SAM 2 conditions each frame on up to 7
  remembered frames, so per-frame cost grows over the first ~7 frames and then
  stays flat. The median is the steady state.
- During the tiny/MPS/fp32 run, a pytest run competed for the GPU for about
  14 s. Its median matches small/MPS/fp32, so the effect looks small.

**What the numbers say.**

- **Memory attention dominates, not the image encoder.** That is why tiny and
  small cost about the same. Profiling tiny on MPS in steady state: in fp32,
  memory attention takes ~0.88 s per frame and the image encoder ~0.23 s. In
  bf16, they take ~0.29 s and ~0.20 s.
- **In fp32, MPS gives no speedup over CPU.** Each memory-attention layer
  cross-attends 4096 queries to ~28.7k memory keys. That single
  `scaled_dot_product_attention` call takes 181 ms in fp32 on MPS, 84 ms on
  CPU, and 53 ms in bf16 or fp16 on MPS. It runs 4 times per frame.
- **So bf16 autocast is the default on MPS.** It is 2.3× faster than fp32, at
  about 0.5 s per frame (1.9 fps for tiny, 1.8 fps for small). On 12 bedroom
  frames the masks agree with fp32 at IoU ≥ 0.999, and the candidate scores
  differ by < 0.01. Half precision is much slower on CPU (7.6 s for that
  attention call), so CPU stays fp32.
- **This is offline, not real time.** A 50-frame sequence takes about 27 s with
  tiny/bf16 on MPS. With `CachedSegmenter`, the Q4 harness pays that once per
  (sequence, model, rule, prompts, erosion, precision).

The only further speedups would change the model itself: fewer memory frames
(`num_maskmem`), or a lower `image_size`. Q4 would then have to grade those as
separate configurations, so none are applied here.

## Tests

`tests/test_segment.py` tests the selection rule (part vs whole, leak rejected
by negatives, ties, fallback, clipping), the prompts, erosion, the cache (hits,
key sensitivity, a fresh process reading the same file) and the protocol. It
uses a fake segmenter, needs neither SAM 2 nor weights, and runs in well under
a second.

One more test runs tiny on three synthetic corridor frames. It is skipped
unless SAM 2 and the tiny weights are present **and** `VORONAV_SAM2=1` is set,
because it takes 5–15 s:

```bash
VORONAV_SAM2=1 .venv/bin/python -m pytest -q tests/test_segment.py
```
