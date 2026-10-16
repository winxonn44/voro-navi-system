import numpy as np
import pytest

from voronav.corruption import (
    CORRUPTIONS,
    boundary_noise,
    boundary_shift,
    mask_error_stats,
    region_dropout,
    signed_floor_distance,
    thin_obstacle_miss,
)
from voronav.maps import random_obstacles

LEVELS = {
    "boundary_shift": 2.0,
    "region_dropout": 0.3,
    "thin_obstacle_miss": 6.0,
    "boundary_noise": 2.0,
}


@pytest.fixture(scope="module")
def occ():
    return random_obstacles(96, 96, 20, seed=4)


def square(n=40, lo=15, hi=25):
    """Free n x n map with a (hi - lo) px square obstacle in the middle."""
    g = np.zeros((n, n), dtype=bool)
    g[lo:hi, lo:hi] = True
    return g


def test_registry_keys():
    assert set(CORRUPTIONS) == set(LEVELS)


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_level_zero_is_identity_copy(occ, name):
    out = CORRUPTIONS[name](occ, 0.0, np.random.default_rng(0))
    assert out.dtype == bool and out.shape == occ.shape
    assert np.array_equal(out, occ)
    assert out is not occ


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_input_not_mutated_and_border_kept(occ, name):
    before = occ.copy()
    out = CORRUPTIONS[name](occ, LEVELS[name], np.random.default_rng(0))
    assert np.array_equal(occ, before)
    assert out.shape == occ.shape and out.dtype == bool
    assert out[0].all() and out[-1].all() and out[:, 0].all() and out[:, -1].all()


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_seeded_reproducible(occ, name):
    a = CORRUPTIONS[name](occ, LEVELS[name], np.random.default_rng(7))
    b = CORRUPTIONS[name](occ, LEVELS[name], np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_signed_floor_distance_half_pixel():
    sd = signed_floor_distance(square())
    assert sd[14, 20] == 0.5 and sd[15, 20] == -0.5 and sd[17, 20] == -2.5


# boundary_shift ---------------------------------------------------------


def test_boundary_shift_exact_square():
    g = square()
    rng = np.random.default_rng(0)
    # Floor grows 2 px into the square: a 10 px square becomes 6 px.
    assert np.array_equal(boundary_shift(g, 2.0, rng), square(lo=17, hi=23))
    # Floor shrinks 2 px: the square is dilated by a Euclidean disc of r = 2.
    yy, xx = np.mgrid[0:40, 0:40]
    dy = np.maximum(np.maximum(15 - yy, yy - 24), 0)
    dx = np.maximum(np.maximum(15 - xx, xx - 24), 0)
    assert np.array_equal(boundary_shift(g, -2.0, rng), dy**2 + dx**2 <= 4)


def test_boundary_shift_monotonic(occ):
    rng = np.random.default_rng(0)
    levels = [-4, -2.5, -1, 0, 1, 2.5, 4]
    outs = [boundary_shift(occ, lv, rng) for lv in levels]
    for a, b in zip(outs, outs[1:]):
        assert np.all(b <= a)  # more floor as level grows: obstacles only shrink
    assert np.all(outs[0] >= occ) and np.all(outs[-1] <= occ)


# region_dropout ---------------------------------------------------------


@pytest.mark.parametrize("level", [0.1, 0.3, 0.6])
def test_region_dropout_fraction(occ, level):
    out = region_dropout(occ, level, np.random.default_rng(1))
    assert np.all(out >= occ)  # only removes floor
    removed = (out & ~occ).sum() / (~occ).sum()
    assert abs(removed - level) < 0.05


def test_region_dropout_full_and_seed_dependence(occ):
    assert region_dropout(occ, 1.0, np.random.default_rng(0)).all()
    a = region_dropout(occ, 0.3, np.random.default_rng(1))
    b = region_dropout(occ, 0.3, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_region_dropout_removes_blobs(occ):
    from scipy.ndimage import label

    out = region_dropout(occ, 0.3, np.random.default_rng(3))
    _, n = label(out & ~occ)
    assert n <= 20  # a few chunks, not salt-and-pepper


# thin_obstacle_miss -----------------------------------------------------


def test_thin_obstacle_miss_pole_vs_wall():
    g = np.zeros((40, 40), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    g[10:12, 10:12] = True  # 2 px pole
    g[20:22, 4:20] = True  # 2 px railing, long but thin
    g[5:35, 25:35] = True  # 10 px wall
    out = thin_obstacle_miss(g, 4.0, np.random.default_rng(0))
    assert not out[10:12, 10:12].any()
    assert not out[20:22, 4:20].any()
    assert out[5:35, 25:35].all()
    assert out[0].all() and out[-1].all() and out[:, 0].all() and out[:, -1].all()
    assert np.all(out <= g)


def test_thin_obstacle_miss_keeps_thin_border_attached():
    g = np.zeros((20, 20), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    g[1:10, 10] = True  # 1 px stub touching the frame
    assert np.array_equal(thin_obstacle_miss(g, 5.0, np.random.default_rng(0)), g)


# boundary_noise ---------------------------------------------------------


@pytest.mark.parametrize("level", [1.0, 3.0])
def test_boundary_noise_within_band(occ, level):
    out = boundary_noise(occ, level, np.random.default_rng(5))
    changed = out != occ
    assert changed.any()
    sd = signed_floor_distance(occ)
    assert np.abs(sd[changed]).max() <= level
    # both directions
    assert (occ & ~out).any() and (~occ & out).any()


def test_boundary_noise_correlated(occ):
    """Errors come in runs along the boundary, not isolated pixels."""
    from scipy.ndimage import label

    changed = boundary_noise(occ, 2.0, np.random.default_rng(5)) != occ
    _, n = label(changed, structure=np.ones((3, 3)))
    assert changed.sum() / n > 4


# mask_error_stats -------------------------------------------------------


def test_stats_identity(occ):
    s = mask_error_stats(occ, occ)
    assert s["free_iou"] == 1.0
    assert s["false_free_frac"] == 0.0 and s["false_obstacle_frac"] == 0.0
    assert s["boundary_error_px"] == 0.0


def test_stats_hand_built():
    t = square()  # 100 obstacle px, 1500 floor px
    e = square(lo=16, hi=24)  # floor grew 1 px: 36 obstacle px lost
    s = mask_error_stats(t, e)
    assert s["false_free_frac"] == pytest.approx(36 / 100)
    assert s["false_obstacle_frac"] == 0.0
    assert s["free_iou"] == pytest.approx(1500 / 1536)
    assert s["boundary_error_px"] == pytest.approx(1.0)
    assert s["n_false_free"] == 36

    e2 = t.copy()
    e2[0:10, 0:10] = True  # 100 floor px blocked
    s2 = mask_error_stats(t, e2)
    assert s2["false_obstacle_frac"] == pytest.approx(100 / 1500)
    assert s2["false_free_frac"] == 0.0
    assert s2["free_iou"] == pytest.approx(1400 / 1500)
    # Estimated boundary adds the block's inner edge (row 9 / col 9, 19 px);
    # brute-force its distance to the true square contour.
    tb = [(y, x) for y in range(15, 25) for x in range(15, 25) if y in (15, 24) or x in (15, 24)]
    eb = tb + [(9, x) for x in range(10)] + [(y, 9) for y in range(9)]
    tb = np.array(tb, float)
    d = [np.hypot(*(tb - p).T).min() for p in np.array(eb, float)]
    assert s2["boundary_error_px"] == pytest.approx(np.mean(d))
