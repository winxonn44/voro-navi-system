"""GVD roadmap properties on maps where the right answer is known."""

import numpy as np
import pytest
from scipy.ndimage import label

from voronav.maps import corridor, random_obstacles
from voronav.planning import build_roadmap
from voronav.planning.roadmap import connect_pieces, obstacle_ids

EIGHT = np.ones((3, 3), dtype=bool)


def test_corridor_skeleton_is_the_centerline():
    occ = corridor(41, 120, gap=15)  # free rows 13..27, center row 20
    rm = build_roadmap(occ)
    rows = np.unique(np.nonzero(rm.skeleton)[0])
    assert set(rows) <= {20, 21}  # 2-px thick by construction (both sides marked)
    assert rm.skeleton[20].all() or rm.skeleton[21].all()
    assert rm.n_pruned == 0


def test_two_rooms_and_a_door():
    # two rooms joined by a door: the roadmap must pass through the door
    occ = np.ones((40, 81), dtype=bool)
    occ[2:-2, 2:39] = False
    occ[2:-2, 42:-2] = False
    occ[17:23, 39:42] = False  # 6-px door
    rm = build_roadmap(occ)
    assert rm.skeleton[17:23, 39:42].any()
    assert label(rm.skeleton, structure=EIGHT)[1] == 1


@pytest.mark.parametrize("seed", range(4))
def test_one_skeleton_piece_per_free_region(seed):
    occ = random_obstacles(128, 128, 12, seed=seed)
    rm = build_roadmap(occ, robot_radius=2.0)
    regions, n_regions = label(rm.safe, structure=EIGHT)
    for r in range(1, n_regions + 1):
        inside = rm.skeleton & (regions == r)
        assert label(inside, structure=EIGHT)[1] <= 1
    assert (rm.skeleton <= rm.safe).all()
    # repair should only drop fragments and add short bridges
    assert rm.n_pruned <= 0.15 * (rm.n_nodes + rm.n_pruned)
    assert rm.bridge_px <= 0.15 * rm.n_nodes


def test_skeleton_respects_robot_radius():
    occ = random_obstacles(96, 96, 8, seed=5)
    rm = build_roadmap(occ, robot_radius=4.0)
    assert (rm.clearance[rm.skeleton] >= 4.0).all()


def test_obstacle_ids_split_border_attached_obstacles():
    occ = np.zeros((20, 20), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[1:6, 4:7] = True   # touches the top wall
    occ[14:19, 12:15] = True  # touches the bottom wall
    ids = obstacle_ids(occ)
    assert ids[3, 5] != ids[16, 13]  # separate obstacles, not merged by the frame
    assert ids[0, 5] == ids[3, 5]    # the frame pixel it touches joins it


def test_connect_pieces_bridges_and_drops_fragments():
    safe = np.ones((10, 40), dtype=bool)
    safe[:, 20] = False  # two regions
    skel = np.zeros_like(safe)
    skel[5, 1:10] = True     # big piece, left region
    skel[5, 13:17] = True    # fragment (4 px), left region: dropped
    skel[2, 22:32] = True    # piece, right region
    skel[8, 33:39] = True    # second sizable piece, right region: bridged
    out, stats = connect_pieces(skel, safe, min_piece_px=5)
    assert stats["n_pruned"] == 4 and not out[5, 13:17].any()
    assert stats["n_bridges"] == 1 and stats["bridge_px"] > 0
    right = out & (np.arange(40) > 20)[None, :]
    assert label(right, structure=EIGHT)[1] == 1
    assert label(out & (np.arange(40) < 20)[None, :], structure=EIGHT)[1] == 1


def test_empty_map_has_no_roadmap():
    rm = build_roadmap(np.zeros((16, 16), dtype=bool))
    assert not rm.skeleton.any() and np.isinf(rm.clearance).all()
