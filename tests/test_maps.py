import numpy as np
import pytest

from voronav.config import MapConfig
from voronav.maps import (
    boundary_sites,
    build_map,
    corridor,
    load_image_mask,
    load_movingai,
    random_obstacles,
    random_sites,
)

MOVINGAI_SAMPLE = """type octile
height 4
width 5
map
@@@@@
@..T@
@.GS@
@@W@@
"""


def test_random_sites_count_and_determinism():
    a = random_sites(32, 48, 20, seed=3)
    b = random_sites(32, 48, 20, seed=3)
    c = random_sites(32, 48, 20, seed=4)
    assert a.shape == (32, 48) and a.dtype == bool
    assert a.sum() == 20
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_sites_rejects_too_many():
    with pytest.raises(ValueError):
        random_sites(2, 2, 5)


def test_random_obstacles_has_border_and_free_space():
    occ = random_obstacles(64, 64, 10, seed=0)
    assert occ[0].all() and occ[-1].all() and occ[:, 0].all() and occ[:, -1].all()
    assert 0 < (~occ).sum() < occ.size


def test_corridor_geometry():
    occ = corridor(20, 30, gap=6)
    free_rows = np.flatnonzero(~occ[:, 0])
    assert len(free_rows) == 6
    assert free_rows[0] == 7 and free_rows[-1] == 12
    assert np.array_equal(occ, np.repeat(occ[:, :1], 30, axis=1))


def test_load_movingai(tmp_path):
    p = tmp_path / "tiny.map"
    p.write_text(MOVINGAI_SAMPLE)
    occ = load_movingai(p)
    expected = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],  # T (tree) blocks
            [1, 0, 0, 0, 1],  # G, S passable
            [1, 1, 1, 1, 1],  # W (water) blocks
        ],
        dtype=bool,
    )
    assert np.array_equal(occ, expected)


@pytest.mark.parametrize(
    "text",
    [
        "type octile\nheight 2\nwidth 2\n..\n..\n",          # no 'map' line
        "type octile\nwidth 2\nmap\n..\n..\n",               # no height
        "type octile\nheight 3\nwidth 2\nmap\n..\n..\n",     # too few rows
        "type octile\nheight 1\nwidth 2\nmap\n.?\n",         # unknown terrain
    ],
)
def test_load_movingai_rejects_malformed(tmp_path, text):
    p = tmp_path / "bad.map"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_movingai(p)


def test_load_image_mask(tmp_path):
    from PIL import Image

    img = np.zeros((8, 10), dtype=np.uint8)
    img[2:6, 3:8] = 255  # bright = free
    Image.fromarray(img).save(tmp_path / "m.png")
    occ = load_image_mask(tmp_path / "m.png")
    assert occ.sum() == 80 - 20
    assert not occ[2:6, 3:8].any()


def test_boundary_sites_are_obstacles_touching_free():
    occ = np.ones((7, 7), dtype=bool)
    occ[2:5, 2:5] = False
    b = boundary_sites(occ)
    assert (b <= occ).all()  # subset of obstacles
    # a ring of 12 pixels 4-adjacent to the 3x3 hole (corners are not)
    assert b.sum() == 12
    assert not b[1, 1] and b[1, 2] and not b[0, 3]


def test_build_map_dispatch():
    assert build_map(MapConfig(kind="points", height=16, width=16, n_sites=5)).sum() == 5
    assert build_map(MapConfig(kind="corridor", height=16, width=16)).any()
    with pytest.raises(ValueError):
        MapConfig(kind="movingai")  # path required
