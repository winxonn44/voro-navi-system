"""Cross-language parity of the roadmap: the C++ core (checkpoint C5) against
roadmap.py, exactly, with no tolerance. Every `Roadmap` field is compared with
array_equal or ==, on the 94 maps of the Q2 benchmark and on random small maps
with random parameters. SciPy's `label` and `minimum_spanning_tree`, which the
C++ reimplements, are checked directly first.

Skipped when the core is not installed. The downloaded maps are skipped when
their folders are missing (CI has only the 24 synthetic maps).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from voronav.config import load_q2_config
from voronav.map_sources import expand, two_rooms
from voronav.maps import boundary_sites, corridor, random_obstacles
from voronav.planning.grid_search import reachable_components, source_field
from voronav.planning.roadmap import (
    build_roadmap,
    connect_pieces,
    gvd_mask,
    obstacle_ids,
)
from voronav.voronoi import distance_sq, jfa_numpy, init_nearest, step_schedule

core = pytest.importorskip("voronav_core")

REPO = Path(__file__).resolve().parents[1]
BENCH_CONFIG = REPO / "configs" / "bench_q2.yaml"
EIGHT = np.ones((3, 3), dtype=bool)
EXPECTED_MAPS = {"synthetic": 24, "movingai_dir": 50, "houseexpo_dir": 20}


def _random_mask(rng, max_side=40):
    h, w = (int(v) for v in rng.integers(1, max_side, size=2))
    return rng.random((h, w)) < float(rng.choice([0.1, 0.3, 0.45, 0.6, 0.8]))


def _blobby_map(rng, max_side=60):
    """Rooms and pillars, which give roadmaps with many pieces."""
    h, w = (int(v) for v in rng.integers(4, max_side, size=2))
    occ = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(0, 12))):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        dy, dx = int(rng.integers(1, max(2, h // 3))), int(rng.integers(1, max(2, w // 3)))
        occ[y : y + dy, x : x + dx] = True
    if rng.random() < 0.7:
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


def _assert_roadmap_equal(ref, out):
    """`ref` from Python, `out` the dict from voronav_core.build_roadmap."""
    for name in ("nearest", "clearance", "safe", "skeleton"):
        a, b = getattr(ref, name), out[name]
        assert b.dtype == a.dtype, name
        assert b.shape == a.shape, name
        assert np.array_equal(b, a), (name, np.argwhere(b != a)[:5])
    assert out["bridges"].dtype == ref.bridges.dtype
    assert np.array_equal(out["bridges"], ref.bridges)
    for name in ("n_pruned", "n_bridges", "bridge_px", "robot_radius"):
        assert out[name] == getattr(ref, name), name


# ---------------------------------------------------------------------------
# What the C++ reimplements from SciPy
# ---------------------------------------------------------------------------


def test_label_matches_scipy_including_numbering():
    rng = np.random.default_rng(0)
    for _ in range(600):
        mask = _random_mask(rng)
        for connectivity, structure in ((4, None), (8, EIGHT)):
            ref, n = label(mask, structure=structure)
            out, count = core.label_components(mask, connectivity)
            assert count == n
            assert np.array_equal(out, ref), (mask.shape, connectivity)


def test_reachable_components_matches_python():
    rng = np.random.default_rng(1)
    for _ in range(300):
        mask = _random_mask(rng)
        ref, n = reachable_components(mask)
        out, count = core.reachable_components(mask)
        assert count == n and np.array_equal(out, ref)
        # corner_ok a superset of the mask, as the roadmap passes it
        corner_ok = mask | (rng.random(mask.shape) < 0.4)
        ref, n = reachable_components(mask, corner_ok)
        out, count = core.reachable_components(mask, corner_ok)
        assert count == n and np.array_equal(out, ref)


def test_reachable_components_is_not_eight_connectivity():
    """dao/lak303d crashed a real planner: diagonal pinches do not connect."""
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert core.reachable_components(mask)[1] == 2
    assert core.label_components(mask, 8)[1] == 1


def _scipy_tree_edges(n, u, v, w):
    tree = minimum_spanning_tree(coo_matrix((w, (u, v)), shape=(n, n))).tocoo()
    return sorted(zip(tree.row.tolist(), tree.col.tolist()))


@pytest.mark.parametrize("seed", range(40))
def test_minimum_spanning_tree_matches_scipy_with_ties(seed):
    """Few distinct weights, so nearly every comparison is a tie. Edges are
    sorted by (row, col), as connect_pieces gives them to SciPy."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    pairs = sorted(
        {
            (int(min(a, b)), int(max(a, b)))
            for a, b in rng.integers(0, n, size=(int(n * 3) + 1, 2))
            if a != b
        }
    )
    if not pairs:
        pytest.skip("no edges drawn")
    u = [p[0] for p in pairs]
    v = [p[1] for p in pairs]
    w = (rng.integers(1, 4, size=len(pairs))).astype(np.float64)
    accepted = core.minimum_spanning_tree(n, u, v, w.tolist())
    assert [(u[i], v[i]) for i in accepted] == _scipy_tree_edges(n, u, v, w)


def test_minimum_spanning_tree_rejects_what_scipy_means_as_no_edge():
    with pytest.raises(ValueError):
        core.minimum_spanning_tree(3, [0], [1], [0.0])
    with pytest.raises(ValueError):
        core.minimum_spanning_tree(3, [0, 1], [1, 2], [1.0, float("nan")])
    with pytest.raises(ValueError, match="sorted"):
        core.minimum_spanning_tree(3, [1, 0], [2, 1], [1.0, 1.0])


# ---------------------------------------------------------------------------
# The roadmap, function by function
# ---------------------------------------------------------------------------


def test_boundary_sites_matches_python():
    rng = np.random.default_rng(2)
    for _ in range(300):
        occ = _random_mask(rng)
        out = core.boundary_sites(occ)
        assert out.dtype == np.bool_
        assert np.array_equal(out, boundary_sites(occ))


def test_obstacle_ids_matches_python():
    rng = np.random.default_rng(3)
    for _ in range(300):
        occ = _random_mask(rng)
        if rng.random() < 0.5 and min(occ.shape) >= 3:  # walled: every frame branch
            occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        assert np.array_equal(core.obstacle_ids(occ), obstacle_ids(occ)), occ.shape


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 7), (7, 1), (2, 2), (2, 9), (9, 2), (3, 3), (3, 8), (8, 3)]
)
def test_obstacle_ids_small_shapes(shape):
    rng = np.random.default_rng(shape)
    for _ in range(100):
        occ = rng.random(shape) < 0.5
        assert np.array_equal(core.obstacle_ids(occ), obstacle_ids(occ)), occ.tolist()


def _voronoi_fields(occ, variant="jfa+2"):
    nearest = jfa_numpy.run_steps(init_nearest(boundary_sites(occ)), step_schedule(*occ.shape, variant))
    return nearest, distance_sq(nearest)


def test_gvd_mask_matches_python():
    rng = np.random.default_rng(4)
    for _ in range(200):
        occ = _blobby_map(rng)
        nearest, d2 = _voronoi_fields(occ)
        angle = float(rng.choice([10.0, 30.0, 60.0, 90.0, 120.0, 180.0]))
        slack = float(rng.choice([0.0, 0.5, 2.0, 5.0]))
        out = core.gvd_mask(occ, nearest, d2, angle, slack)
        assert out.dtype == np.bool_
        assert np.array_equal(out, gvd_mask(occ, nearest, d2, angle, slack)), (occ.shape, angle, slack)


def _assert_source_field_equal(passable, labels, mult):
    g, parent, owner, (cu, cv, cost) = source_field(passable, labels, mult)
    g2, parent2, owner2, (cu2, cv2, cost2) = core.source_field(passable, labels, mult)
    assert np.array_equal(g2, g)
    assert np.array_equal(parent2, parent)
    assert np.array_equal(owner2, owner)
    assert np.array_equal(cu2, cu)
    assert np.array_equal(cv2, cv)
    assert np.array_equal(cost2, cost)


@pytest.mark.parametrize("seed", range(30))
def test_source_field_matches_python_exactly(seed):
    """Parents, owners and contacts, on tie-heavy unit-cost grids and on
    weighted ones, compared bit for bit."""
    rng = np.random.default_rng(100 + seed)
    h, w = (int(v) for v in rng.integers(2, 80, size=2))
    passable = rng.random((h, w)) < 0.75
    labels = np.zeros((h, w), dtype=np.int64)
    n_sources = int(rng.integers(1, 20))
    labels[rng.integers(0, h, n_sources), rng.integers(0, w, n_sources)] = rng.integers(1, 6, n_sources)
    _assert_source_field_equal(passable, labels, None)
    # many pixels tied at a few multiplier values, then irrational ones
    mult = rng.choice([1.0, 1.5, 3.0], size=(h, w))
    _assert_source_field_equal(passable, labels, mult)
    _assert_source_field_equal(passable, labels, 1.0 + 64.0 / rng.uniform(1.0, 9.0, size=(h, w)))


def test_source_field_rejects_bad_multipliers_like_python():
    passable = np.ones((3, 3), dtype=bool)
    labels = np.zeros((3, 3), dtype=np.int64)
    labels[0, 0] = 1
    mult = np.ones((3, 3))
    mult[1, 1] = 0.5
    with pytest.raises(ValueError):
        source_field(passable, labels, mult)
    with pytest.raises(ValueError):
        core.source_field(passable, labels, mult)


@pytest.mark.parametrize("seed", range(40))
def test_connect_pieces_matches_python(seed):
    """Random fragments in random free space, not just real skeletons."""
    rng = np.random.default_rng(200 + seed)
    h, w = (int(v) for v in rng.integers(4, 90, size=2))
    safe = rng.random((h, w)) < 0.85
    skeleton = safe & (rng.random((h, w)) < float(rng.choice([0.02, 0.05, 0.1, 0.3])))
    clearance = rng.uniform(1.0, 12.0, size=(h, w))
    for min_piece_px, lam in ((1, 0.0), (3, 0.0), (8, 0.0), (2, 16.0), (5, 64.0)):
        ref, ref_stats = connect_pieces(skeleton, safe, min_piece_px, clearance, lam)
        out, stats = core.connect_pieces(skeleton, safe, min_piece_px, clearance, lam)
        assert out.dtype == np.bool_
        assert np.array_equal(out, ref)
        assert np.array_equal(stats.pop("bridges"), ref_stats.pop("bridges"))
        assert stats == ref_stats


def test_connect_pieces_rejects_bad_lambda_like_python():
    mask = np.ones((3, 3), dtype=bool)
    for lam, clearance in ((-1.0, None), (1.0, None)):
        with pytest.raises(ValueError):
            connect_pieces(mask, mask, 8, clearance, lam)
        with pytest.raises(ValueError):
            core.connect_pieces(mask, mask, 8, clearance, lam)


def test_build_roadmap_matches_python_on_random_maps():
    rng = np.random.default_rng(300)
    bridges = pruned = 0
    for _ in range(800):
        occ = _blobby_map(rng)
        kwargs = dict(
            robot_radius=float(rng.choice([0.0, 1.0, 1.5, 2.0, 3.0])),
            min_angle_deg=float(rng.choice([30.0, 60.0, 90.0, 150.0])),
            raster_slack_px=float(rng.choice([0.0, 2.0, 4.0])),
            min_piece_px=int(rng.choice([1, 3, 8, 20])),
            bridge_lam=float(rng.choice([0.0, 1.0, 16.0, 64.0])),
        )
        variant = str(rng.choice(["jfa", "jfa+1", "jfa+2", "1+jfa"]))
        ref = build_roadmap(occ, backend="numba", variant=variant, **kwargs)
        out = core.build_roadmap(occ, variant=variant, **kwargs)
        _assert_roadmap_equal(ref, out)
        bridges += ref.n_bridges
        pruned += ref.n_pruned
    assert bridges > 100 and pruned > 100  # the repair paths really ran


def _degenerate_maps():
    occ_all = np.ones((5, 6), dtype=bool)
    free = np.zeros((5, 6), dtype=bool)
    one = np.zeros((9, 11), dtype=bool)
    one[4, 5] = True
    corner = np.zeros((6, 6), dtype=bool)
    corner[0, 0] = True
    line = np.zeros((1, 30), dtype=bool)
    line[0, 7] = line[0, 20] = True
    return [occ_all, free, one, corner, line, np.ones((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool), np.zeros((30, 1), dtype=bool),
            corridor(40, 80, 10), two_rooms(60, 100)]


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("variant", ["jfa", "jfa+1", "jfa+2", "1+jfa"])
def test_build_roadmap_degenerate_maps(index, variant):
    occ = _degenerate_maps()[index]
    ref = build_roadmap(occ, backend="numpy", variant=variant, robot_radius=2.0, bridge_lam=64.0)
    out = core.build_roadmap(occ, variant=variant, robot_radius=2.0, bridge_lam=64.0)
    _assert_roadmap_equal(ref, out)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0)])
def test_empty_grids_raise_like_python(shape):
    occ = np.zeros(shape, dtype=bool)
    with pytest.raises(ValueError):
        build_roadmap(occ, backend="numpy")
    with pytest.raises(ValueError):
        core.build_roadmap(occ)


# ---------------------------------------------------------------------------
# The 94 maps of the Q2 benchmark
# ---------------------------------------------------------------------------


def _bench():
    cfg = load_q2_config(BENCH_CONFIG)
    return cfg, cfg.q2.maps


def _sources(kind):
    cfg, specs = _bench()
    if kind == "synthetic":
        return cfg, [s for s in specs if s.kind not in ("movingai_dir", "houseexpo_dir")], True
    picked = []
    for s in specs:
        if s.kind == kind:
            root = REPO / s.dir
            if not root.is_dir() or not any(root.rglob("*")):
                pytest.skip(f"{kind}: no maps at {root} (downloaded data, see docs/DATA.md)")
            picked.append(replace(s, dir=str(root)))
    assert picked
    return cfg, picked, False


@pytest.mark.parametrize("bridge_lam", ["config", 0.0], ids=["bench-settings", "lam0-ties"])
@pytest.mark.parametrize("kind", ["synthetic", "movingai_dir", "houseexpo_dir"])
def test_benchmark_maps_match_python(kind, bridge_lam):
    """Every field, exactly, on the Q2 benchmark's maps, with the settings it
    uses. bridge_lam=0 is run too: unit step costs make ties everywhere in
    Dijkstra and the spanning tree."""
    cfg, specs, _ = _sources(kind)
    r = cfg.roadmap
    lam = r.bridge_lam if bridge_lam == "config" else bridge_lam
    assert r.variant == "jfa+2" and cfg.q2.robot_radius == 2.0
    params = dict(
        robot_radius=cfg.q2.robot_radius,
        min_angle_deg=r.min_angle_deg,
        raster_slack_px=r.raster_slack_px,
        min_piece_px=r.min_piece_px,
        bridge_lam=lam,
    )
    compared = 0
    for spec in specs:
        for m in expand(spec):
            ref = build_roadmap(m.occ, backend=r.backend, variant=r.variant, **params)
            out = core.build_roadmap(m.occ, variant=r.variant, threads=None, **params)
            try:
                _assert_roadmap_equal(ref, out)
            except AssertionError as e:
                raise AssertionError(f"{m.name}: {e}") from e
            compared += 1
    assert compared == EXPECTED_MAPS[kind]
    print(f"{kind}: {compared} maps compared exactly, bridge_lam={lam}")


# ---------------------------------------------------------------------------
# Determinism and the Python switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [2, 3, 7, 10, None])
@pytest.mark.parametrize("schedule", ["static", "dynamic"])
@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_build_roadmap_is_deterministic_across_threads(threads, schedule, chunk):
    occ = random_obstacles(256, 256, 25, seed=3)
    params = dict(robot_radius=2.0, bridge_lam=64.0)
    serial = core.build_roadmap(occ, **params)
    out = core.build_roadmap(occ, threads=threads, schedule=schedule, chunk=chunk, **params)
    for name in ("nearest", "clearance", "safe", "skeleton", "bridges"):
        assert np.array_equal(out[name], serial[name]), name
    for name in ("n_pruned", "n_bridges", "bridge_px"):
        assert out[name] == serial[name], name


def test_builder_switch_returns_the_same_roadmap():
    occ = two_rooms(60, 100)
    params = dict(robot_radius=2.0, bridge_lam=64.0, variant="jfa+2")
    ref = build_roadmap(occ, backend="numpy", **params)
    out = build_roadmap(occ, builder="cpp", **params)
    assert out.occ is occ
    for name in ("nearest", "clearance", "safe", "skeleton", "bridges"):
        a, b = getattr(ref, name), getattr(out, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert out.n_nodes == ref.n_nodes
    assert (out.n_pruned, out.n_bridges, out.bridge_px) == (ref.n_pruned, ref.n_bridges, ref.bridge_px)
    assert out.robot_radius == ref.robot_radius and out.build_s > 0
    assert isinstance(out.n_bridges, int)


def test_builder_switch_validates():
    occ = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="builder"):
        build_roadmap(occ, builder="fortran")
    with pytest.raises(ValueError, match="CPU"):
        build_roadmap(occ, builder="cpp", device="mps")
