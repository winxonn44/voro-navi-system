import json

import numpy as np
import pytest

from voronav.config import MapConfig
from voronav.maps import (
    boundary_sites,
    build_map,
    corridor,
    load_houseexpo,
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


def _houseexpo(tmp_path, verts, name="h.json"):
    # Same fields as a real HouseExpo file; the loader only reads `verts`.
    xs, ys = [v[0] for v in verts], [v[1] for v in verts]
    doc = {
        "id": "test",
        "room_num": 1,
        "bbox": {"min": [min(xs), min(ys)], "max": [max(xs), max(ys)]},
        "verts": verts,
        "room_category": {"bedroom": [[min(xs), min(ys), max(xs), max(ys)]]},
    }
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_load_houseexpo_square(tmp_path):
    # 1 m square at 4 px/m: a 4x4 free block inside a 2-px obstacle frame.
    p = _houseexpo(tmp_path, [[3.0, 5.0], [4.0, 5.0], [4.0, 6.0], [3.0, 6.0]])
    occ = load_houseexpo(p, px_per_meter=4)
    expected = np.ones((8, 8), dtype=bool)
    expected[2:6, 2:6] = False
    assert occ.dtype == bool
    assert np.array_equal(occ, expected)


def test_load_houseexpo_orientation_and_winding(tmp_path):
    # L-shaped house: 2 m wide along x, 1 m tall, plus a 1x1 m wing at
    # x in [0, 1], y in [1, 2]. Rows follow +y, columns +x.
    ccw = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    occ = load_houseexpo(_houseexpo(tmp_path, ccw), px_per_meter=2, pad=0)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(occ, expected)
    cw = _houseexpo(tmp_path, ccw[::-1], name="cw.json")
    assert np.array_equal(load_houseexpo(cw, px_per_meter=2, pad=0), expected)


def test_load_houseexpo_keeps_thin_interior_wall(tmp_path):
    # Two 1 m rooms joined only through a 0.5 m door: the contour wraps a
    # 0.5 m wall stub hanging down from the top edge at x in [1, 1.5].
    verts = [
        [0, 0], [1, 0], [1, 1.5], [1.5, 1.5], [1.5, 0], [2.5, 0],
        [2.5, 2], [0, 2],
    ]
    occ = load_houseexpo(_houseexpo(tmp_path, verts), px_per_meter=4, pad=1)
    assert occ.shape == (10, 12)
    wall = occ[1:7, 5:7]  # y in [0, 1.5), x in [1, 1.5)
    door = occ[7:9, 5:7]  # y in [1.5, 2)
    assert wall.all() and not door.any()
    assert not occ[1:9, 1:5].any() and not occ[1:9, 7:11].any()


def test_load_houseexpo_default_resolution_fits_budget(tmp_path):
    # A 25 m x 12 m house (large for HouseExpo) stays within ~512 px.
    p = _houseexpo(tmp_path, [[0, 0], [25, 0], [25, 12], [0, 12]])
    occ = load_houseexpo(p)
    assert max(occ.shape) <= 512
    assert occ.shape == (12 * 20 + 4, 25 * 20 + 4)
    assert (~occ).sum() == 25 * 20 * 12 * 20


@pytest.mark.parametrize(
    "doc",
    [{"id": "x"}, {"verts": [[0, 0], [1, 1]]}, {"verts": [1, 2, 3]}],
)
def test_load_houseexpo_rejects_malformed(tmp_path, doc):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_houseexpo(p)


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
