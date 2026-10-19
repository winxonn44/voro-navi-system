"""Q4: config validation, ego-motion warp, the leak gate, the GT control,
and a debug-config smoke run of the harness with a fake segmenter."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from q4_synth import make_sequence
from voronav import corruption
from voronav.config import load_q4_config
from voronav.vision.bev import fov_mask, mask_to_bev
from voronav.vision.dataset import load_sequence
from voronav.vision.egomotion import (
    AlignedTemporalFilter,
    ground_poses,
    leak_score,
    relative_transform,
    warp_bev,
)

ROOT = Path(__file__).resolve().parent.parent
PPM, EXT = 20.0, (5.0, 8.0)
SHAPE = (100, 160)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bench_q4.yaml", "debug_q4.yaml"])
def test_shipped_configs_load(name):
    cfg = load_q4_config(ROOT / "configs" / name)
    assert cfg.q4.filter.gates and cfg.q4.models


@pytest.mark.parametrize("bad", [
    "q4: {models: [huge]}",
    "q4: {rules: [biggest]}",
    "q4: {erosion_px: [0, 0]}",
    "q4: {erosion_px: [1.5]}",
    "q4: {goal_dist_m: [3, 2]}",
    "q4: {filter: {gates: [vote]}}",
    "q4: {filter: {window: 0}}",
    "q4: {inject: {bad: [[melt, 2]]}}",
    "q4: {planners: [rrt_star]}",
    "q4: {typo: 1}",
    "q4: {max_frames: 1}",
])
def test_config_rejects(tmp_path, bad):
    p = tmp_path / "c.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_q4_config(p)


# ---------------------------------------------------------------------------
# Ego-motion warp
# ---------------------------------------------------------------------------


def pattern(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(SHAPE) < 0.3


def test_warp_identity():
    g = pattern()
    p = np.array([3.0, -1.0, 0.7])
    assert np.array_equal(warp_bev(g, p, p, PPM, EXT), g)


def test_warp_forward_motion_shifts_rows_exactly():
    g = pattern()
    # robot drives 1 m north, heading north: everything comes 20 rows closer
    w = warp_bev(g, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), PPM, EXT)
    assert np.array_equal(w[20:], g[:-20])
    assert w[:20].all()  # never seen by the old frame -> obstacle/unseen


def test_warp_lateral_motion_shifts_cols_exactly():
    g = pattern(1)
    # robot heading east (yaw 90 deg) moves 0.5 m south = to its right:
    # the scene appears 10 columns further left
    src, dst = np.array([0.0, 0.0, np.pi / 2]), np.array([-0.5, 0.0, np.pi / 2])
    w = warp_bev(g, src, dst, PPM, EXT)
    assert np.array_equal(w[:, :-10], g[:, 10:])


def test_warp_rotation_moves_a_point_exactly():
    # one obstacle cell ~2 m ahead; after turning 90 deg left in place it
    # is ~2 m to the right
    g = np.zeros(SHAPE, bool)
    g[59, 79] = True  # X = 5 - 59.5/20 = 2.025, Y = 4 - 79.5/20 = 0.025
    src = np.array([0.0, 0.0, 0.0])          # facing north
    dst = np.array([0.0, 0.0, -np.pi / 2])   # facing west (yaw runs north -> east)
    A, b = relative_transform(src, dst)
    assert np.allclose(A @ np.array([0.025, -2.025]) + b, [2.025, 0.025])
    w = warp_bev(g, src, dst, PPM, EXT, fill=False)
    assert w.sum() == 1
    assert tuple(np.argwhere(w)[0]) == (99, 120)  # X = 0.025, Y = -2.025


def test_ground_poses_yaw():
    q = lambda yaw: [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    poses = np.array([[1, 2, -0.6, *q(0.3)], [4, 5, -0.6, *q(-2.0)]])
    gp = ground_poses(poses)
    assert np.allclose(gp, [[1, 2, 0.3], [4, 5, -2.0]])


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def room_grid():
    """A seen wedge-free test grid: a room with a pillar, everything seen."""
    occ = np.ones(SHAPE, bool)
    occ[10:95, 20:140] = False
    occ[40:50, 70:80] = True
    return occ


def test_leak_gate_flags_leaks_the_iou_gate_misses():
    occ = room_grid()
    seen = np.ones(SHAPE, bool)
    rng = np.random.default_rng(0)
    leak = corruption.boundary_shift(occ, 3, rng)
    p = np.zeros(3)
    # a 3 px leak barely moves the IoU...
    from voronav.vision.egomotion import aligned_iou
    assert aligned_iou(leak, occ, seen) > 0.8
    # ...but adds new free space along the whole boundary
    assert leak_score(leak, occ, seen, 3.0) > 0.05
    assert leak_score(occ, occ, seen, 3.0) == 0.0
    for gate, expect in (("iou", "accept"), ("leak", "reject"), ("both", "reject")):
        f = AlignedTemporalFilter(PPM, EXT, window=3, gate=gate)
        f.step(occ, seen, p)
        f.step(occ, seen, p)
        out, st = f.step(leak, seen, p)
        assert st.gate == expect, gate
        if expect == "reject":
            assert np.array_equal(out, occ)  # the leak never reaches the output


def test_leak_gate_quiet_under_pure_motion():
    """Clean frames of a moving camera must not look like leaks once aligned."""
    occ = room_grid()
    seen = np.ones(SHAPE, bool)
    f = AlignedTemporalFilter(PPM, EXT, window=5, gate="both")
    for t in range(6):
        pose = np.array([0.05 * t, 0.0, 0.0])
        cur = warp_bev(occ, np.zeros(3), pose, PPM, EXT)
        cur_seen = warp_bev(seen, np.zeros(3), pose, PPM, EXT, fill=False)
        _, st = f.step(cur, cur_seen, pose)
        assert st.gate in ("first", "accept"), (t, st)


def test_filter_absorbs_one_dropout_frame():
    occ = room_grid()
    seen = np.ones(SHAPE, bool)
    drop = corruption.region_dropout(occ, 0.5, np.random.default_rng(1))
    f = AlignedTemporalFilter(PPM, EXT, window=5, gate="iou")
    for _ in range(3):
        f.step(occ, seen, np.zeros(3))
    out, st = f.step(drop, seen, np.zeros(3))
    assert st.gate == "reject" and np.array_equal(out, occ)


def test_filter_only_frees_cells_in_view():
    occ = room_grid()
    seen = np.zeros(SHAPE, bool)
    seen[:, :80] = True
    f = AlignedTemporalFilter(PPM, EXT, window=3, gate="iou")
    out, _ = f.step(occ, seen, np.zeros(3))
    assert out[:, 80:].all()


# ---------------------------------------------------------------------------
# Harness pieces on a synthetic sequence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("q4")
    make_sequence(root, n=8)
    return root


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_q4
    return bench_q4


def test_synth_sequence_loads(synth):
    seq = load_sequence(synth / "Synth_P0000")
    assert np.allclose(seq.camera_height_m, 0.6, atol=1e-3)
    gt = mask_to_bev(seq.floor_masks[0], seq.intrinsics, seq.camera(0))
    assert 0.05 < (~gt).mean() < 0.6


def test_gt_control_never_collides(synth, bench):
    cfg = load_q4_config(ROOT / "configs" / "debug_q4.yaml")
    seq = load_sequence(synth / "Synth_P0000")
    gt = bench.sequence_truth(cfg, seq, len(seq))
    specs = bench.planner_specs(cfg)
    n = 0
    for t in range(len(seq)):
        assert gt["start"][t] is not None and gt["goals"][t]
        _, res = bench.plan_frame(cfg, specs, gt["occ"][t], gt["start"][t], gt["goals"][t],
                                  gt["clr"][t], gt["safe"][t])
        for out in res.values():
            assert not any(out["collision"]) and not any(out["margin_violation"])
            assert not any(out["stranded"])
            n += len(out["collision"])
    assert n == len(seq) * len(specs) * cfg.q4.goals_per_frame


def test_gt_stream_through_aligned_vote_stays_close(synth, bench):
    """Warping exact GT grids and voting reproduces the current GT grid up
    to a resampling ring (the warp error budget)."""
    cfg = load_q4_config(ROOT / "configs" / "debug_q4.yaml")
    seq = load_sequence(synth / "Synth_P0000")
    gt = bench.sequence_truth(cfg, seq, len(seq))
    st = bench.streams(cfg, gt["occ"], gt["seen"], gt["gposes"])
    for t, g in enumerate(st["vote_iou"][0]):
        s = corruption.mask_error_stats(gt["occ"][t], g)
        assert s["free_iou"] > 0.9, t


def test_debug_smoke_run(tmp_path, synth):
    out = tmp_path / "run"
    cmd = [sys.executable, "scripts/bench_q4.py", "--config", "configs/debug_q4.yaml",
           "--out", str(out), "--data", str(synth), "--fake-segmenter"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    kinds = {r["row"] for r in rows}
    assert kinds == {"sequence", "sam", "frame"}
    frames = [r for r in rows if r["row"] == "frame"]
    conds = {r["cond"] for r in frames}
    assert conds == {"gt", "inject:boundary_shift=3", "sam2.1-tiny/largest_consistent/e0",
                     "sam2.1-tiny/largest_consistent/e2"}
    # 4 conditions x 3 streams x 8 frames
    assert len(frames) == 4 * 3 * 8
    for r in frames:
        if r["cond"] == "gt" and r["stream"] == "raw":
            assert all(not any(v["collision"]) for v in r["out"].values())
            assert r["bev_free_iou"] == 1.0
        if r["source"] == "sam":
            assert 0 < r["img_free_iou"] <= 1 and r["lat"]["sam_s"] > 0
        assert len(r["signed_hist"]) == 21
    # the fake leaks every 4th frame: erosion must cut its false floor
    ff = {e: np.mean([r["img_false_free_frac"] for r in frames
                      if r.get("erode_px") == e and r["stream"] == "raw"]) for e in (0, 2)}
    assert ff[2] < ff[0]
    # rerun hits the mask cache
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    sam = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()
           if '"row": "sam"' in l]
    assert all(r["cache_hit"] for r in sam)

    subprocess.run([sys.executable, "scripts/plot_q4.py", str(out / "results.jsonl"),
                    "--out", str(tmp_path / "figs")], cwd=ROOT, check=True, capture_output=True)
    assert any((tmp_path / "figs").glob("q4_*.png"))
    res = subprocess.run([sys.executable, "scripts/summarize_q4.py", str(out / "results.jsonl")],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    assert "# Q4 summary" in res.stdout


# ---------------------------------------------------------------------------
# Demo pieces (no SAM: the GIF itself needs weights)
# ---------------------------------------------------------------------------


def test_demo_future_cell_and_goal():
    from voronav.demo import future_cell, pick_goal, plan_frame
    poses = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # 2 m straight ahead
    r, c = future_cell(poses, 0, 1)
    assert (r, c) == pytest.approx((59.5, 79.5))
    occ = room_grid()
    p = plan_frame(occ, target_rc=(r, c))
    assert p["start"] is not None and p["goal"] is not None
    assert p["vor"] is not None and p["astar"] is not None
    safe = ~occ
    assert pick_goal(safe, p["start"]) is not None


def test_cli_help(capsys):
    from voronav.cli import main
    main(["--help"])
    assert "demo" in capsys.readouterr().out
