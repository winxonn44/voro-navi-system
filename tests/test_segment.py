"""Floor segmentation interface: selection rule, prompts, erosion, caching.

Everything except the last test runs without SAM 2 or its weights.
"""

import importlib.util
import os

import numpy as np
import pytest

from voronav.vision.segment import (
    SAM2_DIR,
    SAM2_MODELS,
    CachedSegmenter,
    FloorSegmenter,
    Prompts,
    default_floor_prompts,
    erode,
    select_mask,
)

H, W = 40, 60


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


# --- prompts -----------------------------------------------------------------

def test_prompts_normalise_and_round_trip():
    p = Prompts(positive=[[1, 2], (3, 4)], negative=[(5, 6)], box=[0, 0, 10, 10], frames=[0, 3])
    assert p.positive == ((1.0, 2.0), (3.0, 4.0))
    assert p.frames == (0, 3)
    assert Prompts.from_json(p.to_json()) == p
    assert hash(p) == hash(Prompts.from_json(p.to_json()))
    coords, labels = p.labelled_points()
    assert coords.shape == (3, 2) and coords.dtype == np.float32
    assert labels.tolist() == [1, 1, 0]


def test_prompts_need_something_to_go_on():
    with pytest.raises(ValueError):
        Prompts(positive=())
    with pytest.raises(ValueError):
        Prompts(positive=((1, 1),), frames=())
    with pytest.raises(ValueError):
        Prompts(positive=((1, 1),), box=(0, 0, 1))
    Prompts(positive=(), box=(0, 0, 5, 5))  # a box alone is enough


def test_default_prompts_low_positives_high_negatives():
    p = default_floor_prompts(480, 640, frames=(0, 10))
    assert p.frames == (0, 10)
    assert len(p.positive) == 3 and len(p.negative) == 2
    for x, y in p.positive:
        assert 0 <= x < 640 and y > 0.8 * 480
    for x, y in p.negative:
        assert y < 0.2 * 480
    assert {x < 320 for x, _ in p.negative} == {True, False}  # both corners


# --- multimask selection -----------------------------------------------------

FLOOR = default_floor_prompts(H, W)
WHOLE = rect(20, 40, 0, 60)    # the whole floor: bottom half
PART = rect(30, 40, 20, 40)    # a rug: contains the centre click only
LEAK = rect(0, 40, 0, 60)      # everything, including both upper corners


def test_top_score_is_sams_choice():
    masks = np.stack([PART, WHOLE])
    assert select_mask(masks, np.array([0.95, 0.80]), FLOOR, "top_score") == 0


def test_largest_consistent_prefers_whole_floor_over_part():
    # the part has the higher score but misses the side clicks
    masks = np.stack([PART, WHOLE])
    assert select_mask(masks, np.array([0.95, 0.80]), FLOOR) == 1


def test_largest_consistent_rejects_leak_through_negatives():
    masks = np.stack([WHOLE, LEAK, PART])
    assert select_mask(masks, np.array([0.7, 0.9, 0.95]), FLOOR) == 0


def test_largest_consistent_ties_go_to_score():
    other = rect(20, 40, 0, 60)
    masks = np.stack([WHOLE, other])
    assert select_mask(masks, np.array([0.6, 0.9]), FLOOR) == 1
    assert select_mask(masks, np.array([0.9, 0.6]), FLOOR) == 0


def test_largest_consistent_falls_back_to_fewest_violations():
    # nothing is consistent: PART misses 2 positives, LEAK hits 2 negatives,
    # this one misses 1 positive; it wins despite the lowest score
    almost = rect(20, 40, 0, 35)
    masks = np.stack([PART, LEAK, almost])
    assert select_mask(masks, np.array([0.9, 0.9, 0.1]), FLOOR) == 2


def test_select_mask_validates():
    with pytest.raises(ValueError):
        select_mask(np.stack([WHOLE]), np.array([0.5, 0.5]), FLOOR)
    with pytest.raises(ValueError):
        select_mask(np.stack([WHOLE]), np.array([0.5]), FLOOR, "biggest")


def test_points_outside_image_are_clipped():
    p = Prompts(positive=((-5, 100),))  # clips to the bottom-left pixel
    masks = np.stack([rect(0, 5, 0, 5), rect(35, 40, 0, 5)])
    assert select_mask(masks, np.array([0.9, 0.1]), p) == 1


# --- erosion -------------------------------------------------------------------

def test_erode_zero_is_identity_copy():
    out = erode(WHOLE, 0)
    assert np.array_equal(out, WHOLE) and out is not WHOLE


def test_erode_shrinks_only_and_keeps_image_border_floor():
    out = erode(WHOLE, 3)
    assert out.dtype == bool and out.shape == WHOLE.shape
    assert not (out & ~WHOLE).any()          # never adds floor
    assert not out[20:23].any() and out[23:].all()  # the wall edge moved 3 px
    assert out[-1].all()                     # the bottom edge did not


def test_erode_removes_thin_slivers():
    m = WHOLE.copy()
    m[5:20, 30] = True                       # a 1 px leak up the wall
    assert not erode(m, 1)[5:19, 30].any()


# --- protocol and cache ------------------------------------------------------

class FakeSegmenter:
    """Floor = dark pixels; counts calls so caching is observable."""

    cache_tag = "fake/v1"

    def __init__(self):
        self.video_calls = 0

    def segment_image(self, frame, prompts):
        return frame.mean(axis=2) < 128

    def segment_video(self, frames, prompts):
        self.video_calls += 1
        return [self.segment_image(f, prompts) for f in frames]


def frames(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(n)]


def test_fake_satisfies_protocol():
    seg = FakeSegmenter()
    assert isinstance(seg, FloorSegmenter)
    out = seg.segment_video(frames(), FLOOR)
    assert len(out) == 4 and all(m.dtype == bool and m.shape == (H, W) for m in out)


def test_cache_hits_and_round_trips(tmp_path):
    inner = FakeSegmenter()
    seg = CachedSegmenter(inner, tmp_path)
    assert isinstance(seg, FloorSegmenter)
    fr = frames()
    a = seg.segment_video(fr, FLOOR, sequence="hall-01")
    b = seg.segment_video(fr, FLOOR, sequence="hall-01")
    assert inner.video_calls == 1 and (seg.hits, seg.misses) == (1, 1)
    assert all(x.dtype == bool and np.array_equal(x, y) for x, y in zip(a, b))
    assert len(list(tmp_path.glob("*.npz"))) == 1
    # a fresh wrapper (new process) reads the same file
    again = CachedSegmenter(FakeSegmenter(), tmp_path)
    again.segment_video(fr, FLOOR, sequence="hall-01")
    assert again.inner.video_calls == 0


def test_cache_key_depends_on_sequence_prompts_and_model(tmp_path):
    seg = CachedSegmenter(FakeSegmenter(), tmp_path)
    fr = frames()
    base = seg.key(fr, FLOOR, "hall-01")
    assert seg.key(fr, FLOOR, "hall-02") != base
    assert seg.key(fr, default_floor_prompts(H, W, frames=(1,)), "hall-01") != base
    assert seg.key(fr[:3], FLOOR, "hall-01") != base

    class Other(FakeSegmenter):
        cache_tag = "fake/v2"

    assert CachedSegmenter(Other(), tmp_path).key(fr, FLOOR, "hall-01") != base


def test_cache_without_sequence_hashes_frames(tmp_path):
    seg = CachedSegmenter(FakeSegmenter(), tmp_path)
    assert seg.key(frames(seed=0), FLOOR, None) == seg.key(frames(seed=0), FLOOR, None)
    assert seg.key(frames(seed=0), FLOOR, None) != seg.key(frames(seed=1), FLOOR, None)


def test_cache_rejects_bad_frames(tmp_path):
    seg = CachedSegmenter(FakeSegmenter(), tmp_path)
    with pytest.raises(ValueError):
        seg.segment_video([np.zeros((H, W), np.uint8)], FLOOR)
    with pytest.raises(ValueError):
        seg.segment_video([np.zeros((H, W, 3), np.float32)], FLOOR)
    with pytest.raises(ValueError):
        seg.segment_video([], FLOOR)


# --- optional: the real model ----------------------------------------------
# Takes ~5-15 s (model build + 4 encoder passes), so it is opt-in:
#   VORONAV_SAM2=1 .venv/bin/python -m pytest -q tests/test_segment.py

_HAVE_SAM2 = importlib.util.find_spec("sam2") is not None
_HAVE_TINY = (SAM2_DIR / SAM2_MODELS["tiny"][1]).exists()
_OPTED_IN = os.environ.get("VORONAV_SAM2") == "1"


@pytest.mark.skipif(not (_HAVE_SAM2 and _HAVE_TINY), reason="needs SAM 2 + tiny weights")
@pytest.mark.skipif(not _OPTED_IN, reason="slow; set VORONAV_SAM2=1 to run")
def test_sam2_tiny_on_synthetic_corridor():
    from voronav.vision.segment import Sam2Floor

    h, w = 96, 128
    rng = np.random.default_rng(0)
    fr = []
    for t in range(3):
        f = np.empty((h, w, 3), np.uint8)
        f[:] = (200, 190, 170)                       # walls
        f[h // 2 :] = (70, 70, 80)                   # floor, lower half
        f = np.clip(f + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)
        f[h // 2 - 2 : h // 2 + 2] = (30, 25, 20)    # skirting board
        fr.append(np.roll(f, t, axis=1))
    seg = Sam2Floor("tiny", erode_px=2)
    out = seg.segment_video(fr, default_floor_prompts(h, w))
    assert len(out) == 3
    for m in out:
        assert m.dtype == bool and m.shape == (h, w)
        assert m[int(0.75 * h) :].mean() > 0.8      # floor found
        assert m[: int(0.4 * h)].mean() < 0.05      # walls not floor
    assert len(seg.last_choices) == 1 and len(seg.last_choices[0]["scores"]) == 3
