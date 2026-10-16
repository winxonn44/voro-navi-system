"""The Q2 map-source registry: specs expand to the right named maps, and
directory sources load every matching file."""

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import label

from voronav.config import MapSourceConfig
from voronav.map_sources import KINDS, expand, expand_all, two_rooms


def test_synthetic_expands_sizes_times_seeds():
    spec = MapSourceConfig("random_obstacles", sizes=[32, 48], seeds=[0, 1, 2], n_shapes=4)
    maps = list(expand(spec))
    assert [m.name for m in maps][:2] == ["random_obstacles-32-s0", "random_obstacles-32-s1"]
    assert len(maps) == 6 and len({m.name for m in maps}) == 6
    assert {m.occ.shape for m in maps} == {(32, 32), (48, 48)}
    assert all(m.occ.dtype == bool for m in maps)
    # seeds change the map, the same seed reproduces it
    again = list(expand(spec))
    assert not np.array_equal(maps[0].occ, maps[1].occ)
    assert np.array_equal(maps[0].occ, again[0].occ)


def test_deterministic_kinds_one_map_per_size():
    maps = list(expand_all([MapSourceConfig("corridor", sizes=[40], gap=8),
                            MapSourceConfig("two_rooms", sizes=[60, 120])]))
    assert [m.name for m in maps] == ["corridor-40", "two_rooms-60", "two_rooms-120"]
    assert maps[2].occ.shape == (120, 200)


@pytest.mark.parametrize("size", [60, 120, 240])
def test_two_rooms_door_joins_the_rooms(size):
    occ = two_rooms(size, round(size * 5 / 3))
    _, n = label(~occ, structure=np.ones((3, 3), dtype=bool))
    assert n == 1  # one free region: the door is open at every scale


def test_directory_sources(tmp_path):
    (tmp_path / "b.map").write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\n...\n")
    (tmp_path / "a.map").write_text("type octile\nheight 1\nwidth 2\nmap\n.@\n")
    (tmp_path / "notes.txt").write_text("ignored")
    maps = list(expand(MapSourceConfig("movingai_dir", dir=str(tmp_path))))
    assert [m.name for m in maps] == ["movingai_dir-a", "movingai_dir-b"]
    assert maps[1].occ.tolist() == [[False, True, False], [False, False, False]]
    assert maps[0].params["file"].endswith("a.map")
    assert len(list(expand(MapSourceConfig("movingai_dir", dir=str(tmp_path), max_maps=1)))) == 1

    img = np.full((4, 5), 255, dtype=np.uint8)
    img[1, 2] = 0
    Image.fromarray(img).save(tmp_path / "room.png")
    (png,) = expand(MapSourceConfig("image_dir", dir=str(tmp_path)))
    assert png.occ.sum() == 1 and png.occ[1, 2]


def test_directory_sources_recurse_into_set_folders(tmp_path):
    # fetch_maps.py stores MovingAI maps one folder per set (movingai/maze/...)
    for set_name, stem in (("maze", "m1"), ("room", "r1"), ("maze", "m2")):
        (tmp_path / set_name).mkdir(exist_ok=True)
        (tmp_path / set_name / f"{stem}.map").write_text(
            "type octile\nheight 1\nwidth 2\nmap\n..\n")
    maps = list(expand(MapSourceConfig("movingai_dir", dir=str(tmp_path))))
    assert [m.name for m in maps] == [
        "movingai_dir-maze-m1", "movingai_dir-maze-m2", "movingai_dir-room-r1"]
    assert [m.params["set"] for m in maps] == ["maze", "maze", "room"]


def test_houseexpo_directory_source(tmp_path):
    verts = [[0, 0], [2, 0], [2, 1], [0, 1]]
    (tmp_path / "h1.json").write_text('{"verts": %s}' % verts)
    (house,) = expand(MapSourceConfig("houseexpo_dir", dir=str(tmp_path)))
    assert house.name == "houseexpo_dir-h1" and (~house.occ).any()


def test_missing_or_empty_directory_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no directory"):
        list(expand(MapSourceConfig("movingai_dir", dir=str(tmp_path / "nope"))))
    with pytest.raises(FileNotFoundError, match="no \\*.png"):
        list(expand(MapSourceConfig("image_dir", dir=str(tmp_path))))


def test_duplicate_names_rejected():
    spec = MapSourceConfig("corridor", sizes=[40], gap=8)
    with pytest.raises(ValueError, match="duplicate"):
        list(expand_all([spec, spec]))


def test_registry_lists_every_kind():
    assert set(KINDS) >= {"random_obstacles", "corridor", "two_rooms", "movingai_dir", "image_dir"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cluttered_room_has_thin_obstacles(seed):
    from scipy.ndimage import binary_opening

    from voronav.map_sources import cluttered_room
    from voronav.planning import build_roadmap
    from voronav.planning.grid_search import reachable_components

    occ = cluttered_room(256, seed)
    assert occ.dtype == bool and occ.shape == (256, 256)
    assert np.array_equal(occ, cluttered_room(256, seed))  # seeded
    # opening with a 5 x 5 square deletes everything at most 4 px thin
    lab, n = label(occ, structure=np.ones((3, 3), dtype=bool))
    opened = binary_opening(occ, np.ones((5, 5), dtype=bool))
    thin = [i for i in range(1, n + 1) if not opened[lab == i].any()]
    assert len(thin) >= 50  # chair legs, poles, railings
    walls = np.bincount(lab.ravel())[1:].argmax() + 1
    assert walls not in thin  # the walls are thick and survive
    # a long thin component: a railing
    assert any((lab == i).sum() >= 30 for i in thin)
    # the robot (radius 2) can reach every room: one search region
    _, regions = reachable_components(build_roadmap(occ, robot_radius=2.0).safe)
    assert regions == 1


def test_cluttered_room_is_a_seeded_kind():
    maps = list(expand(MapSourceConfig("cluttered_room", sizes=[128], seeds=[0, 1])))
    assert [m.name for m in maps] == ["cluttered_room-128-s0", "cluttered_room-128-s1"]
    assert not np.array_equal(maps[0].occ, maps[1].occ)
