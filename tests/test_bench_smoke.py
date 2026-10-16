"""End-to-end: the benchmark harness and plotter run on the debug config and
produce well-formed rows and figures."""

import json
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
