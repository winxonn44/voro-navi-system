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
    # per query: voronoi + astar + 2 lambdas, plus RRT* on queries 0 and 2
    assert len(queries) == 3 * (3 * 4 + 2)
    for q in queries:
        assert q["plan_s"] >= 0 and q["expansions"] >= 0
        assert q["euclid_px"] >= math.hypot(q["height"], q["width"]) / 3 - 1e-9
        assert {"length_px", "min_clearance_px", "mean_clearance_px", "collision_free"} <= set(q)
        if q["planner"] in ("voronoi", "astar", "weighted_astar"):
            # complete planners, endpoints in one safe region
            assert q["success"] and q["collision_free"]
    by_query = {}
    for q in queries:  # every planner saw the same start/goal
        by_query.setdefault((q["map"], q["query"]), set()).add((tuple(q["start"]), tuple(q["goal"])))
    assert all(len(v) == 1 for v in by_query.values())
    assert {q["lam"] for q in queries if q["planner"] == "weighted_astar"} == {4, 32}

    subprocess.run(
        [sys.executable, "scripts/plot_q2.py", str(tmp_path / "results.jsonl"),
         "--out", str(tmp_path / "figs")],
        cwd=ROOT, check=True, capture_output=True,
    )
    for name in ("q2_tradeoff.png", "q2_success.png", "q2_query_time.png",
                 "q2_build_time.png", "q2_amortized.png"):
        assert (tmp_path / "figs" / name).stat().st_size > 0
