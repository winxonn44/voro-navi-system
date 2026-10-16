"""GVD roadmap properties on maps where the right answer is known."""

import numpy as np

from voronav.planning.grid_search import REACHABILITY, reachable_components
import pytest
from scipy.ndimage import label

from voronav.maps import corridor, random_obstacles
from voronav.planning import build_roadmap, exact_clearance
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
    regions, n_regions = label(rm.safe, structure=REACHABILITY)
    for r in range(1, n_regions + 1):
        inside = rm.skeleton & (regions == r)
        assert reachable_components(inside, corner_ok=rm.safe)[1] <= 1
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


def test_clearance_weighted_bridges():
    # Two skeleton pieces; the straight bridge between them grazes a wall
    # stub, a slightly longer one stays in the open.
    safe = np.ones((30, 60), dtype=bool)
    safe[8:15, 28:31] = False  # stub between the pieces' near ends
    clearance = exact_clearance(~safe)
    skel = np.zeros_like(safe)
    skel[11, 5:22] = True
    skel[11, 37:55] = True
    short, s0 = connect_pieces(skel, safe, clearance=clearance)
    wide, s1 = connect_pieces(skel, safe, clearance=clearance, bridge_lam=32.0)
    assert s0["n_bridges"] == s1["n_bridges"] == 1
    assert s1["bridge_px"] >= s0["bridge_px"]
    assert (short | ~s0["bridges"]).all() and s0["bridges"].sum() == s0["bridge_px"]
    assert clearance[s1["bridges"]].min() > clearance[s0["bridges"]].min()
    for out in (short, wide):
        assert label(out, structure=EIGHT)[1] == 1
    with pytest.raises(ValueError):
        connect_pieces(skel, safe, bridge_lam=1.0)  # needs the clearance field


def test_diagonal_pinch_is_not_a_connection():
    # Two open areas touching only at a corner. Searches may not cut corners,
    # so they are separate regions; the repair must not try to bridge them
    # (it raised "skeleton piece unreachable" on MovingAI dao/lak303d).
    occ = np.ones((41, 41), dtype=bool)
    occ[1:20, 1:20] = False   # top-left room
    occ[20:40, 20:40] = False  # bottom-right room, touching at (19,19)/(20,20)
    rm = build_roadmap(occ)    # must not raise
    regions, n = label(rm.safe, structure=REACHABILITY)
    assert n == 2
    for r in (1, 2):
        assert reachable_components(rm.skeleton & (regions == r), corner_ok=rm.safe)[1] == 1


def test_reachable_components_matches_search_rules():
    from voronav.planning import search
    mask = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=bool)
    # corners blocked: three separate pieces, as the search sees it
    assert reachable_components(mask)[1] == 3
    # corners free (as on a skeleton inside open space): one piece
    assert reachable_components(mask, corner_ok=np.ones((3, 3), bool))[1] == 1
    assert search(mask, (0, 0), goal=(2, 2), corner_ok=np.ones((3, 3), bool)).path is not None
    # plain free-space regions equal 4-connectivity
    occ = random_obstacles(64, 64, 10, seed=1)
    a, na = reachable_components(~occ)
    b, nb = label(~occ, structure=REACHABILITY)
    assert na == nb and np.array_equal(a > 0, b > 0)


def test_connect_pieces_many_pieces_one_tree():
    # A dashed skeleton: 40 pieces of 9 px in one room. The repair joins them
    # with exactly 39 bridges (a spanning tree), in one search.
    safe = np.ones((40, 400), dtype=bool)
    skel = np.zeros_like(safe)
    for x0 in range(0, 400, 10):
        skel[20, x0:x0 + 9] = True
    out, stats = connect_pieces(skel, safe)
    assert stats["n_bridges"] == 39 and stats["bridge_px"] == 39
    assert reachable_components(out, corner_ok=safe)[1] == 1
    assert stats["bridges"].sum() == stats["bridge_px"] and not (stats["bridges"] & skel).any()


def test_source_field_labels_nearest_source():
    from voronav.planning.grid_search import source_field
    safe = np.ones((5, 21), dtype=bool)
    labels = np.zeros(safe.shape, dtype=np.int64)
    labels[2, 0], labels[2, 20] = 1, 2
    g, parent, owner, (u, v, cost) = source_field(safe, labels)
    assert owner[2, 5] == 1 and owner[2, 15] == 2 and g[2, 5] == 5.0
    assert (owner > 0).all() and len(u) > 0
    assert (owner.ravel()[u] != owner.ravel()[v]).all()
    assert cost.min() == pytest.approx(20.0)  # the straight join through the middle
