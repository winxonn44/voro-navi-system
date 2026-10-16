import numpy as np
import pytest

from voronav.corruption import CORRUPTIONS
from voronav.maps import random_obstacles
from voronav.temporal import bad_frame_sequence, free_iou, temporal_filter


@pytest.fixture(scope="module")
def occ():
    return random_obstacles(96, 96, 20, seed=4)


def test_sequence_shapes_flags_and_seed(occ):
    frames, flags = bad_frame_sequence(occ, 12, 0.3, np.random.default_rng(0), return_flags=True)
    assert len(frames) == 12 and flags.shape == (12,) and flags.dtype == bool
    assert all(f.shape == occ.shape and f.dtype == bool for f in frames)
    again = bad_frame_sequence(occ, 12, 0.3, np.random.default_rng(0))
    assert all(np.array_equal(a, b) for a, b in zip(frames, again))
    # bad frames (dropout 0.4) are much worse than base frames
    iou = np.array([free_iou(occ, f) for f in frames])
    assert iou[flags].max() < iou[~flags].min()


def test_sequence_extremes_and_any_registry_name(occ):
    rng = np.random.default_rng(1)
    assert all(free_iou(occ, f) > 0.85 for f in bad_frame_sequence(occ, 5, 0.0, rng))
    for name in CORRUPTIONS:
        lvl = 0.3 if name == "region_dropout" else 4.0
        _, flags = bad_frame_sequence(occ, 3, 1.0, rng, bad=(name, lvl), return_flags=True)
        assert flags.all()
    thin = bad_frame_sequence(occ, 2, 1.0, rng, base=None, bad=("thin_obstacle_miss", 6.0))
    assert np.all(thin[0] <= occ)
    with pytest.raises(ValueError):
        bad_frame_sequence(occ, 2, 1.0, rng, bad=("nope", 1.0))


def test_filter_rejects_single_bad_frame_and_recovers(occ):
    rng = np.random.default_rng(2)
    frames = bad_frame_sequence(occ, 10, 0.0, rng)
    bad = CORRUPTIONS["region_dropout"](occ, 0.4, rng)
    frames[5] = bad
    out = temporal_filter(frames)
    assert free_iou(frames[5], frames[4]) < 0.8  # it is an outlier
    assert free_iou(out[5], occ) > 0.9  # ...and it was rejected
    assert free_iou(out[5], out[4]) > 0.95
    assert free_iou(out[9], occ) > 0.9


def test_filter_masks_transient_thin_miss():
    """A single frame that loses thin obstacles passes the IoU gate, but the
    vote outvotes it."""
    occ = np.zeros((40, 40), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[10:12, 10:12] = occ[25:27, 30:32] = True  # two 2 px poles
    frames = [occ.copy() for _ in range(6)]
    frames[4] = CORRUPTIONS["thin_obstacle_miss"](occ, 4.0, np.random.default_rng(0))
    assert not frames[4][10:12, 10:12].any()
    out = temporal_filter(frames)
    assert free_iou(frames[4], frames[3]) >= 0.8
    assert np.array_equal(out[4], occ)


def test_filter_is_causal(occ):
    frames = bad_frame_sequence(occ, 10, 0.3, np.random.default_rng(3))
    out = temporal_filter(frames)
    tampered = frames[:6] + bad_frame_sequence(occ, 4, 1.0, np.random.default_rng(9))
    out2 = temporal_filter(tampered)
    assert all(np.array_equal(a, b) for a, b in zip(out[:6], out2[:6]))


def test_filter_p_bad_zero_changes_little(occ):
    frames = bad_frame_sequence(occ, 12, 0.0, np.random.default_rng(4))
    out = temporal_filter(frames)
    raw = np.mean([free_iou(occ, f) for f in frames])
    filt = np.mean([free_iou(occ, f) for f in out])
    assert filt >= raw - 0.02
    assert all(free_iou(o, f) > 0.85 for o, f in zip(out, frames))


def test_filter_bad_first_frame_and_sustained_failure(occ):
    rng = np.random.default_rng(5)
    good = bad_frame_sequence(occ, 8, 0.0, rng)
    first_bad = [CORRUPTIONS["region_dropout"](occ, 0.4, rng)] + good
    out = temporal_filter(first_bad, patience=3)
    assert free_iou(out[-1], occ) > 0.9  # not locked onto the bad first frame

    sustained = good[:3] + bad_frame_sequence(occ, 8, 1.0, rng)
    out = temporal_filter(sustained, patience=3)
    assert free_iou(out[-1], occ) < 0.8  # a sustained failure is not hidden


def test_filter_window_one_and_identity(occ):
    frames = [occ.copy() for _ in range(4)]
    assert all(np.array_equal(o, occ) for o in temporal_filter(frames, window=1))
    assert all(np.array_equal(o, occ) for o in temporal_filter(frames))
    before = [f.copy() for f in frames]
    temporal_filter(frames)
    assert all(np.array_equal(a, b) for a, b in zip(frames, before))
