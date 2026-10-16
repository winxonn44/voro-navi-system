"""End-to-end: the benchmark harness and plotter run on the debug config and
produce well-formed rows and figures."""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_bench_and_plot_end_to_end(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/bench_jfa.py", "--config", "configs/debug.yaml",
         "--out", str(tmp_path)],
        cwd=ROOT, check=True, capture_output=True,
    )
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    # 2 sizes x 1 density x 2 variants x (numpy + numba t1,t2 + torch cpu)
    assert len(rows) == 2 * 1 * 2 * 4
    for r in rows:
        assert r["median_s"] > 0 and len(r["times_s"]) == 2
        assert 0.0 <= r["error_rate"] < 0.05
    # error is a property of the variant, not the backend
    by_key = {}
    for r in rows:
        by_key.setdefault((r["size"], r["variant"]), set()).add(r["error_rate"])
    assert all(len(v) == 1 for v in by_key.values())

    subprocess.run(
        [sys.executable, "scripts/plot_q1.py", str(tmp_path / "results.jsonl"),
         "--out", str(tmp_path / "figs")],
        cwd=ROOT, check=True, capture_output=True,
    )
    for name in ("q1_throughput.png", "q1_scaling.png", "q1_error.png"):
        assert (tmp_path / "figs" / name).stat().st_size > 0


def test_q2_bench_and_plot_end_to_end(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/bench_plan.py", "--config", "configs/debug_q2.yaml",
         "--out", str(tmp_path)],
        cwd=ROOT, check=True, capture_output=True,
    )
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    maps = [r for r in rows if r["row"] == "map"]
    queries = [r for r in rows if r["row"] == "query"]
    # 2 random maps + 1 two_rooms, 3 queries each
    assert [m["map"] for m in maps] == ["random_obstacles-64-s0", "random_obstacles-64-s1",
                                        "two_rooms-60"]
    for m in maps:
        assert m["build_s"] > 0 and m["n_nodes"] > 0 and m["n_queries"] == 3
        assert {"n_bridges", "bridge_px", "n_pruned", "pixels", "machine"} <= set(m)
    # per query: 2 voronoi variants + astar + 2 lambdas, plus RRT* on queries 0 and 2
    assert len(queries) == 3 * (3 * 5 + 2)
    for q in queries:
        assert q["plan_s"] >= 0 and q["expansions"] >= 0
        assert q["euclid_px"] >= math.hypot(q["height"], q["width"]) / 3 - 1e-9
        assert {"length_px", "min_clearance_px", "mean_clearance_px", "collision_free"} <= set(q)
        if q["planner"] in ("voronoi", "astar", "weighted_astar"):
            # complete planners, endpoints in one safe region
            assert q["success"] and q["collision_free"]
        for mode in ("safe", "preserve"):  # smoothed metrics on every planner's row
            if q["success"]:
                assert q[f"{mode}_collision_free"]
                assert q[f"{mode}_length_px"] <= q["length_px"] + 1e-6
    by_query = {}
    for q in queries:  # every planner saw the same start/goal
        by_query.setdefault((q["map"], q["query"]), set()).add((tuple(q["start"]), tuple(q["goal"])))
    assert all(len(v) == 1 for v in by_query.values())
    assert {q["lam"] for q in queries if q["planner"] == "weighted_astar"} == {4, 32}
    assert {q["variant"] for q in queries if q["planner"] == "voronoi"} == {"nearest", "best"}
    assert {q["variant"] for q in queries if q["planner"] != "voronoi"} == {None}

    subprocess.run(
        [sys.executable, "scripts/plot_q2.py", str(tmp_path / "results.jsonl"),
         "--out", str(tmp_path / "figs")],
        cwd=ROOT, check=True, capture_output=True,
    )
    for name in ("q2_tradeoff.png", "q2_tradeoff_safe.png", "q2_tradeoff_preserve.png",
                 "q2_success.png", "q2_query_time.png", "q2_build_time.png", "q2_amortized.png"):
        assert (tmp_path / "figs" / name).stat().st_size > 0


def test_q3_bench_and_plot_end_to_end(tmp_path):
    import pytest

    pytest.importorskip("voronav.corruption")  # the error models
    pytest.importorskip("voronav.temporal")
    subprocess.run(
        [sys.executable, "scripts/bench_q3.py", "--config", "configs/debug_q3.yaml",
         "--out", str(tmp_path)],
        cwd=ROOT, check=True, capture_output=True,
    )
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["row"], []).append(r)
    assert [m["map"] for m in by["map"]] == ["random_obstacles-64-s0", "cluttered_room-96-s0",
                                             "cluttered_room-96-s1"]
    # 2 maps x (3 boundary_shift + 2 thin_obstacle_miss levels) x 1 draw
    assert len(by["corrupt"]) == 2 * 5
    # 2 voronoi variants + astar + 1 lambda, 3 queries per corrupted map
    assert len(by["query"]) == 2 * 5 * 3 * 4
    for q in by["query"]:
        assert {"stranded", "collision", "margin_violation", "free_iou",
                "false_free_frac", "min_clearance_px", "tag"} <= set(q)
        if q["collision"]:
            assert q["margin_violation"]  # every collision is a violation
        if q["stranded"]:
            assert q["strand_reason"] in ("endpoint", "no_path") and q["length_px"] is None
        if q["level"] == 0:  # the control: the true map
            assert not q["stranded"] and not q["margin_violation"]
            assert q["free_iou"] == 1.0 and q["in_true_safe"]
    for c in by["corrupt"]:
        if c["level"] == 0:
            assert c["identical"]
    # cluttered_room really loses obstacles to thin_obstacle_miss
    assert any(c["false_free_frac"] > 0 for c in by["corrupt"]
               if c["map"].startswith("cluttered") and c["kind"] == "thin_obstacle_miss")
    # Part B: 1 map, 2 queries x 4 planners, 2 p_bad x 2 streams x 4 frames
    assert len(by["frame"]) == 2 * 4 * 2 * 2 * 4
    assert len(by["sequence"]) == 2 * 4 * 2 * 2
    assert {s["stream"] for s in by["sequence"]} == {"raw", "filtered"}
    for s in by["sequence"]:
        assert 0 <= s["margin_violation_frac"] <= 1 and 0 <= s["stranded_frac"] <= 1
        if s["p_bad"] == 0:
            assert s["n_bad"] == 0
    assert {r["group"] for r in by["query"]} == {"structural", "cluttered"}
    # one filter trace per sequence (1 map x 1 bad model x 2 p_bad), one entry per frame
    assert len(by["filter"]) == 2
    for flt in by["filter"]:
        assert len(flt["gate"]) == len(flt["is_bad"]) == 4 and flt["gate"][0] == "first"
        assert set(flt["gate"]) <= {"first", "accept", "reject", "reset"}
        assert flt["n_rejected_bad"] <= min(flt["n_rejected"], flt["n_bad"])

    subprocess.run(
        [sys.executable, "scripts/plot_q3.py", str(tmp_path / "results.jsonl"),
         "--out", str(tmp_path / "figs")],
        cwd=ROOT, check=True, capture_output=True,
    )
    for name in ("q3_collision.png", "q3_margin.png", "q3_stranded.png",
                 "q3_min_clearance.png", "q3_frames.png", "q3_filter_detection.png"):
        assert (tmp_path / "figs" / name).stat().st_size > 0


def test_q3_filter_trace_mirrors_temporal_filter():
    """bench_q3 logs the gate's decisions from its own copy of the filter;
    it must produce the same masks as voronav.temporal on hard cases."""
    import numpy as np
    import pytest

    temporal = pytest.importorskip("voronav.temporal")
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_q3 import filter_trace

    from voronav.map_sources import cluttered_room

    occ = cluttered_room(96, 0)
    rng = np.random.default_rng(0)
    for bad in (("region_dropout", 0.4), ("thin_obstacle_miss", 4.0)):
        frames, is_bad = temporal.bad_frame_sequence(occ, 20, 0.5, rng, base=("boundary_noise", 0.5),
                                                     bad=bad, return_flags=True)
        for window, patience in ((5, 3), (3, None), (1, 1)):
            real = temporal.temporal_filter(frames, window=window, outlier_iou=0.9,
                                            patience=patience)
            mine, trace = filter_trace(frames, window, 0.9, patience)
            assert all(np.array_equal(a, b) for a, b in zip(real, mine))
            assert all(len(t["history"]) <= window for t in trace)
