"""Floor segmentation for Q4: a model-agnostic interface and a SAM 2 backend.

A FloorSegmenter turns RGB frames (uint8 [H, W, 3]) into floor masks (bool
[H, W], True = floor), following the conventions in voronav.vision.

Sam2Floor uses SAM 2's video predictor: on each prompt frame it asks the image
decoder for its candidate masks, keeps one by a selection rule, hands that mask
to the video predictor, and propagates it through the sequence with SAM 2's
memory. The selection rule exists because a single click often returns a part
of the floor (a rug, one tile region) instead of all of it; see select_mask.

Nothing here imports torch or sam2 at module load, so the interface, the
selection rule, erosion and caching are usable (and tested) without them.
Install the model with the `vision` extra; see docs/Q4_SAM.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SAM2_DIR = ROOT / "data" / "raw" / "sam2"
CACHE_DIR = ROOT / "runs" / "q4_masks"  # gitignored

# model name -> (hydra config inside the sam2 package, checkpoint file name)
SAM2_MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
}

SelectRule = Literal["top_score", "largest_consistent"]
Precision = Literal["fp32", "bf16", "fp16"]
SELECT_RULES = ("top_score", "largest_consistent")

Point = tuple[float, float]


@dataclass(frozen=True)
class Prompts:
    """Clicks (and an optional box) telling the segmenter where the floor is.

    Points are (x, y) in pixels of the frame they apply to, SAM's convention.
    positive: points on the floor.  negative: points on walls or obstacles.
    box: optional (x0, y0, x1, y1) around the floor.
    frames: the frame indices the prompts are applied on; segment_video
    propagates from these with the model's memory.  segment_image ignores it.
    """

    positive: tuple[Point, ...]
    negative: tuple[Point, ...] = ()
    box: tuple[float, float, float, float] | None = None
    frames: tuple[int, ...] = (0,)

    def __post_init__(self):
        # normalise lists from callers or JSON into hashable tuples
        object.__setattr__(self, "positive", _points(self.positive))
        object.__setattr__(self, "negative", _points(self.negative))
        if self.box is not None:
            object.__setattr__(self, "box", tuple(float(v) for v in self.box))
            if len(self.box) != 4:
                raise ValueError("box must be (x0, y0, x1, y1)")
        object.__setattr__(self, "frames", tuple(int(f) for f in self.frames))
        if not self.positive and self.box is None:
            raise ValueError("need at least one positive point or a box")
        if not self.frames:
            raise ValueError("need at least one prompt frame")

    def to_json(self) -> str:
        """Canonical JSON (used as part of the cache key)."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Prompts":
        return cls(**json.loads(s))

    def labelled_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(coords float32 [N, 2] as x, y; labels int32 [N], 1 = pos, 0 = neg)."""
        pts = self.positive + self.negative
        coords = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        labels = np.array([1] * len(self.positive) + [0] * len(self.negative), np.int32)
        return coords, labels


def _points(pts) -> tuple[Point, ...]:
    return tuple((float(x), float(y)) for x, y in pts)


def default_floor_prompts(h: int, w: int, frames: Sequence[int] = (0,)) -> Prompts:
    """A heuristic default for a forward-facing camera at roughly hip height.

    Positives: three points in the bottom band (centre at 92% height, left and
    right at 85% height, 30% / 70% width), where a camera pointed along the
    floor almost always sees floor right in front of it.  Negatives: the two
    upper corners (8% in from each side, 8% down), usually wall or ceiling.
    This is a guess about framing, not knowledge of the scene: it fails when
    the camera pitches down at an obstacle or up at the ceiling, or when an
    object sits right in front of the robot.  Q4 reports how often it fails.
    """
    return Prompts(
        positive=((0.5 * w, 0.92 * h), (0.3 * w, 0.85 * h), (0.7 * w, 0.85 * h)),
        negative=((0.08 * w, 0.08 * h), (0.92 * w, 0.08 * h)),
        frames=tuple(frames),
    )


def _point_hits(mask: np.ndarray, pts: Sequence[Point]) -> np.ndarray:
    """mask value at each (x, y) point, clipped into the image."""
    if not pts:
        return np.zeros(0, bool)
    h, w = mask.shape
    xy = np.asarray(pts, dtype=float).reshape(-1, 2)
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return mask[y, x]


def select_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    prompts: Prompts,
    rule: SelectRule = "largest_consistent",
) -> int:
    """Pick one of SAM's candidate masks; returns its index.

    masks  bool [K, H, W] candidates; scores [K] SAM's predicted IoU.

    "top_score": SAM's own choice, the highest predicted IoU.
    "largest_consistent": among candidates that contain every positive point
        and no negative point, the one with the most pixels; ties on area go
        to the higher predicted IoU.  If no candidate is consistent, the one
        with the fewest prompt violations wins, then area, then IoU.
    The second rule targets the part-instead-of-whole failure: SAM's top score
    often goes to a tidy sub-region (a rug, a lit patch) rather than the whole
    floor.  Its risk is the opposite error, a mask that leaks into walls, which
    Q3 found is the dangerous direction; negatives and erode_px guard against
    it, and Q4 measures both rules.
    """
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores) or len(masks) == 0:
        raise ValueError("masks must be [K, H, W] with one score per mask, K >= 1")
    if rule == "top_score":
        return int(np.argmax(scores))
    if rule != "largest_consistent":
        raise ValueError(f"unknown rule {rule!r}; expected one of {SELECT_RULES}")
    best, best_key = 0, None
    for k, m in enumerate(masks):
        misses = int((~_point_hits(m, prompts.positive)).sum())
        hits_neg = int(_point_hits(m, prompts.negative).sum())
        key = (-(misses + hits_neg), int(m.sum()), scores[k])
        if best_key is None or key > best_key:
            best, best_key = k, key
    return best


def disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink the floor by px pixels with a disk; px = 0 is a copy.

    Pixels outside the image count as floor (border_value=True), so the floor
    that runs off the bottom edge stays attached to it; only real boundaries
    with walls and obstacles move inward.  Q3 found that a shrunken floor mask
    fails safe while a leaking one causes collisions, so this trades coverage
    for safety.
    """
    mask = np.asarray(mask, dtype=bool)
    if px <= 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=disk(px), border_value=True)


@runtime_checkable
class FloorSegmenter(Protocol):
    """Anything that turns frames into floor masks."""

    def segment_video(
        self, frames: Sequence[np.ndarray], prompts: Prompts
    ) -> list[np.ndarray]:
        """One bool [H, W] mask per frame, prompts applied on prompts.frames."""
        ...

    def segment_image(self, frame: np.ndarray, prompts: Prompts) -> np.ndarray:
        """A bool [H, W] mask for a single frame."""
        ...


def _check_frames(frames: Sequence[np.ndarray]) -> tuple[int, int]:
    if len(frames) == 0:
        raise ValueError("no frames")
    shape = frames[0].shape
    for f in frames:
        if f.dtype != np.uint8 or f.ndim != 3 or f.shape[2] != 3:
            raise ValueError("frames must be uint8 [H, W, 3] RGB")
        if f.shape != shape:
            raise ValueError("all frames must have the same size")
    return shape[0], shape[1]


class CachedSegmenter:
    """Wraps any FloorSegmenter and stores its video masks on disk.

    The key is sequence + the segmenter's cache_tag (model, rule, erosion) +
    prompts + frame count and size.  Without a sequence name the frames' bytes
    are hashed instead.  The device is not part of the key: MPS and CPU masks
    differ by a few boundary pixels at most, and the harness should not redo
    work because it ran on a different backend.  Files are compressed .npz of
    bit-packed masks under cache_dir.
    """

    def __init__(self, inner: FloorSegmenter, cache_dir: Path | str = CACHE_DIR):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    @property
    def cache_tag(self) -> str:
        return _cache_tag(self.inner)

    def key(self, frames: Sequence[np.ndarray], prompts: Prompts, sequence: str | None) -> str:
        h, w = _check_frames(frames)
        if sequence is None:
            fh = hashlib.sha256()
            for f in frames:
                fh.update(np.ascontiguousarray(f).tobytes())
            sequence = "frames-" + fh.hexdigest()[:16]
        blob = json.dumps(
            {"seq": sequence, "seg": self.cache_tag, "prompts": prompts.to_json(),
             "n": len(frames), "hw": [h, w]},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def path(self, frames, prompts, sequence: str | None = None) -> Path:
        return self.cache_dir / f"{self.key(frames, prompts, sequence)}.npz"

    def segment_video(
        self, frames: Sequence[np.ndarray], prompts: Prompts, sequence: str | None = None
    ) -> list[np.ndarray]:
        p = self.path(frames, prompts, sequence)
        if p.exists():
            with np.load(p) as z:
                n, h, w = (int(v) for v in z["shape"])
                bits = np.unpackbits(z["bits"], count=n * h * w)
            self.hits += 1
            return list(bits.reshape(n, h, w).astype(bool))
        masks = [np.asarray(m, dtype=bool) for m in self.inner.segment_video(frames, prompts)]
        stack = np.stack(masks)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, bits=np.packbits(stack.reshape(-1)), shape=np.array(stack.shape),
            meta=np.array(json.dumps({"seq": sequence, "seg": self.cache_tag,
                                      "prompts": prompts.to_json()})),
        )
        tmp.replace(p)
        self.misses += 1
        return masks

    def segment_image(self, frame: np.ndarray, prompts: Prompts) -> np.ndarray:
        return self.inner.segment_image(frame, prompts)


def _cache_tag(seg) -> str:
    return getattr(seg, "cache_tag", type(seg).__name__)


def default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Sam2Floor:
    """Floor segmentation with SAM 2.1 (tiny or small) via its video predictor.

    model: "tiny" | "small".  device: "mps" | "cpu" (None = MPS when present).
    select: how to pick among SAM's candidate masks on prompt frames, see
        select_mask.  erode_px: shrink every output mask by this many pixels.
    precision: "fp32", or "bf16" / "fp16" autocast (MPS only).  On MPS, fp32
        memory attention dominates the per-frame cost (docs/Q4_SAM.md); half
        precision cuts it about 3x.  On CPU half precision is far slower, so
        it is refused there.  None = "bf16" on MPS, "fp32" on CPU.
    checkpoint_dir: where scripts/fetch_sam2.py put the weights.

    SAM 2's optional CUDA extension (hole filling) is absent here, so SAM 2
    skips that step (its warning is silenced).
    """

    def __init__(
        self,
        model: Literal["tiny", "small"] = "tiny",
        device: str | None = None,
        select: SelectRule = "largest_consistent",
        erode_px: int = 0,
        precision: Precision | None = None,
        checkpoint_dir: Path | str = SAM2_DIR,
    ):
        if model not in SAM2_MODELS:
            raise ValueError(f"model must be one of {sorted(SAM2_MODELS)}")
        if select not in SELECT_RULES:
            raise ValueError(f"select must be one of {SELECT_RULES}")
        cfg, ckpt_name = SAM2_MODELS[model]
        ckpt = Path(checkpoint_dir) / ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing; run scripts/fetch_sam2.py")
        try:
            with _quiet():
                from sam2.build_sam import build_sam2_video_predictor
                from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError("SAM 2 is not installed; see docs/Q4_SAM.md") from e

        self.model = model
        self.device = device or default_device()
        self.select = select
        self.erode_px = int(erode_px)
        if precision is None:
            precision = "bf16" if self.device == "mps" else "fp32"
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16 or fp16")
        if precision != "fp32" and self.device != "mps":
            raise ValueError("half precision is only supported on mps")
        self.precision = precision
        with _quiet():
            self.predictor = build_sam2_video_predictor(cfg, str(ckpt), device=self.device)
            # the image predictor shares the same network; it only exposes the
            # decoder's multimask output, which the video predictor hides
            self.image_predictor = SAM2ImagePredictor(self.predictor)
        self.last_choices: list[dict] = []  # per prompt frame: index, scores, areas

    @property
    def cache_tag(self) -> str:
        return f"sam2.1-{self.model}/{self.select}/erode{self.erode_px}/{self.precision}"

    def _autocast(self):
        import torch

        if self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device, dtype=dtype)

    def _candidates(self, frame: np.ndarray, prompts: Prompts) -> tuple[np.ndarray, int]:
        """SAM's candidate masks on one frame and the chosen index."""
        import torch

        coords, labels = prompts.labelled_points()
        with torch.inference_mode(), self._autocast():
            self.image_predictor.set_image(frame)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=coords if len(coords) else None,
                point_labels=labels if len(labels) else None,
                box=None if prompts.box is None else np.asarray(prompts.box, np.float32),
                multimask_output=True,
            )
        masks = masks.astype(bool)
        k = select_mask(masks, scores, prompts, self.select)
        self.last_choices.append({
            "index": k, "scores": [float(s) for s in scores],
            "areas": [int(m.sum()) for m in masks],
        })
        return masks, k

    def segment_image(self, frame: np.ndarray, prompts: Prompts) -> np.ndarray:
        _check_frames([frame])
        self.last_choices = []
        masks, k = self._candidates(frame, prompts)
        return erode(masks[k], self.erode_px)

    def segment_video(
        self, frames: Sequence[np.ndarray], prompts: Prompts
    ) -> list[np.ndarray]:
        import torch

        h, w = _check_frames(frames)
        if any(not 0 <= f < len(frames) for f in prompts.frames):
            raise ValueError("prompt frame index out of range")
        import sam2.sam2_video_predictor as svp

        self.last_choices = []
        with (torch.inference_mode(), self._autocast(), _quiet(),
              _swap(svp, "tqdm", lambda it, **_: it)):
            state = self._init_state(frames)
            for f in sorted(set(prompts.frames)):
                masks, k = self._candidates(frames[f], prompts)
                self.predictor.add_new_mask(state, frame_idx=f, obj_id=1, mask=masks[k])
            out: list[np.ndarray | None] = [None] * len(frames)
            start = min(prompts.frames)
            for idx, _, logits in self.predictor.propagate_in_video(state, start_frame_idx=start):
                out[idx] = (logits[0, 0] > 0).cpu().numpy()
            if start > 0:  # frames before the first prompt: track backwards
                for idx, _, logits in self.predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=True
                ):
                    out[idx] = (logits[0, 0] > 0).cpu().numpy()
            self.predictor.reset_state(state)
        return [erode(m, self.erode_px) for m in out]

    def _init_state(self, frames: Sequence[np.ndarray]) -> dict:
        """SAM 2's init_state, fed from arrays instead of a JPEG folder.

        init_state only accepts a path, so its frame loader is swapped for one
        that does the same preprocessing (PIL resize to image_size, /255,
        ImageNet mean/std) on our arrays; the rest of init_state is untouched.
        """
        import sam2.sam2_video_predictor as svp
        import torch
        from PIL import Image

        h, w = frames[0].shape[:2]

        def load(video_path, image_size, offload_video_to_cpu, compute_device, **_):
            mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            imgs = torch.empty(len(frames), 3, image_size, image_size)
            for i, f in enumerate(frames):
                r = np.array(Image.fromarray(f).resize((image_size, image_size)))
                imgs[i] = torch.from_numpy(r).permute(2, 0, 1).float() / 255.0
            imgs = (imgs - mean) / std
            if not offload_video_to_cpu:
                imgs = imgs.to(compute_device)
            return imgs, h, w

        with _swap(svp, "load_video_frames", load):
            return self.predictor.init_state(video_path="<arrays>")


@contextlib.contextmanager
def _quiet():
    """Silence SAM 2's known-harmless warnings: torch.jit on Python 3.14 and
    the missing CUDA extension (it only skips hole filling)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
        warnings.filterwarnings("ignore", message=".*cannot import name '_C'.*")
        yield


@contextlib.contextmanager
def _swap(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)
