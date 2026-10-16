"""Frame sequences with occasional bad segmentations, and a causal filter.

Models a video stream: most frames are segmented a little imperfectly, a few
are badly wrong (SAM clipped the floor, or merged a pole away). A short
temporal filter should absorb the few bad frames; a sustained failure
should still get through, because hiding it would be worse. In Q4 the
frames come from SAM 2 instead of `bad_frame_sequence`.
"""

from __future__ import annotations

import numpy as np

from .corruption import CORRUPTIONS

__all__ = ["bad_frame_sequence", "free_iou", "temporal_filter"]


def _apply(occ: np.ndarray, spec: tuple[str, float] | None, rng: np.random.Generator):
    if spec is None:
        return occ.copy()
    name, level = spec
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](occ, level, rng)


def bad_frame_sequence(
    occ: np.ndarray,
    n_frames: int,
    p_bad: float,
    rng: np.random.Generator,
    base: tuple[str, float] | None = ("boundary_noise", 1.0),
    bad: tuple[str, float] | None = ("region_dropout", 0.4),
    return_flags: bool = False,
):
    """`n_frames` corrupted copies of `occ`.

    Each frame independently gets the severe `bad` corruption with
    probability `p_bad`, else the mild `base` one; each frame draws fresh
    noise. `base`/`bad` are (registry name, level) pairs, or None for no
    corruption. With `return_flags=True`, returns (frames, is_bad) where
    is_bad is a bool array of length n_frames.
    """
    is_bad = rng.random(n_frames) < p_bad
    frames = [_apply(occ, bad if b else base, rng) for b in is_bad]
    return (frames, is_bad) if return_flags else frames


def free_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of the floor (False) pixels of two occupancy grids."""
    fa, fb = ~a, ~b
    union = int((fa | fb).sum())
    return int((fa & fb).sum()) / union if union else 1.0


def temporal_filter(
    frames: list[np.ndarray],
    window: int = 5,
    outlier_iou: float = 0.8,
    patience: int = 3,
) -> list[np.ndarray]:
    """Causal outlier gate plus majority vote over recent accepted masks.

    For each frame t (seeing only frames <= t):
    1. Gate: if its floor IoU with the last accepted mask is below
       `outlier_iou`, reject it and keep using the last accepted mask.
       The first frame is always accepted.
    2. Vote: output obstacle wherever at least half of the last `window`
       accepted masks say obstacle (ties go to obstacle, the safe side).

    `patience`: after this many rejections in a row the gate assumes the
    world (or the segmenter) really changed, accepts the current frame and
    restarts the history from it. This keeps a bad first frame from locking
    the filter forever, and lets a sustained failure through instead of
    hiding it. None disables the reset.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    history: list[np.ndarray] = []
    rejected = 0
    out = []
    for f in frames:
        if not history:
            history.append(f)
        elif free_iou(f, history[-1]) >= outlier_iou:
            history.append(f)
            rejected = 0
        else:
            rejected += 1
            if patience is not None and rejected >= patience:
                history = [f]
                rejected = 0
        history = history[-window:]
        votes = np.sum(history, axis=0)
        out.append(2 * votes >= len(history))
    return out
