"""Summarize a bench_q3.py results.jsonl into the Markdown tables used in
the README's Q3 section.

    python scripts/summarize_q3.py runs/bench_q3/results.jsonl > docs/q3/q3_summary.md

Part A rates are per (map group, corruption, level, planner) over all
queries. Part B rates are per frame, pooled over planners, raw vs
temporal-filtered. Gate detection is per filter stream (one per map,
query-independent sequence, bad model and p_bad).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TAGS = ["astar", "weighted_astar@16", "weighted_astar@64", "voronoi[nearest]", "voronoi[best64]"]
NAMES = {"astar": "A*", "weighted_astar@16": "wA* λ16", "weighted_astar@64": "wA* λ64",
         "voronoi[nearest]": "Vor nearest", "voronoi[best64]": "Vor best64"}
KINDS = ["boundary_shift", "region_dropout", "thin_obstacle_miss", "boundary_noise"]


def pct(a: int, n: int) -> str:
    return "–" if n == 0 else f"{100 * a / n:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()

    # (group, kind, level, tag) -> [n, collision, margin, stranded]
    a = defaultdict(lambda: [0, 0, 0, 0])
    strand_reason = defaultdict(lambda: defaultdict(int))
    # (bad, p_bad, stream) -> [n, collision, margin, stranded]
    b = defaultdict(lambda: [0, 0, 0, 0])
    b_tag = defaultdict(lambda: [0, 0])  # (bad, p_bad, stream, tag) -> [n, collision]
    det = defaultdict(lambda: [0, 0, 0])  # bad -> [n_bad, n_rejected, n_rejected_bad]
    maps = {"A": set(), "B": set()}
    n_rows = defaultdict(int)

    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            n_rows[r["row"]] += 1
            if r["row"] == "query":
                maps["A"].add(r["map"])
                k = (r["group"] if "group" in r else "all", r["kind"], r["level"], r["tag"])
                s = a[k]
                s[0] += 1
                s[1] += bool(r["collision"])
                s[2] += bool(r["margin_violation"])
                s[3] += bool(r["stranded"])
                if r["stranded"]:
                    strand_reason[r["kind"]][r.get("strand_reason") or "?"] += 1
            elif r["row"] == "frame":
                maps["B"].add(r["map"])
                bad = f"{r['bad'][0]} {r['bad'][1]:g}"
                s = b[(bad, r["p_bad"], r["stream"])]
                s[0] += 1
                s[1] += bool(r["collision"])
                s[2] += bool(r["margin_violation"])
                s[3] += bool(r["stranded"])
                t = b_tag[(bad, r["p_bad"], r["stream"], r["tag"])]
                t[0] += 1
                t[1] += bool(r["collision"])
            elif r["row"] == "filter":
                bad = f"{r['bad'][0]} {r['bad'][1]:g}"
                d = det[bad]
                d[0] += r["n_bad"]
                d[1] += r["n_rejected"]
                d[2] += r["n_rejected_bad"]

    print("# Q3 summary\n")
    print(f"Part A: {len(maps['A'])} maps, {n_rows['query']} planner runs. "
          f"Part B: {len(maps['B'])} maps, {n_rows['frame']} per-frame plans.\n")

    groups = sorted({k[0] for k in a})
    for g in groups:
        print(f"## Part A, {g} maps: collision % (path enters a true obstacle)\n")
        print("| corruption | level | " + " | ".join(NAMES[t] for t in TAGS) + " | stranded % (all planners) |")
        print("|---|---|" + "---|" * len(TAGS) + "---|")
        for kind in KINDS:
            levels = sorted({k[2] for k in a if k[0] == g and k[1] == kind})
            for lv in levels:
                cells = [pct(a[(g, kind, lv, t)][1], a[(g, kind, lv, t)][0]) for t in TAGS]
                n = sum(a[(g, kind, lv, t)][0] for t in TAGS)
                st = sum(a[(g, kind, lv, t)][3] for t in TAGS)
                print(f"| {kind} | {lv:g} | " + " | ".join(cells) + f" | {pct(st, n)} |")
        print()
        print(f"## Part A, {g} maps: margin violation % (true clearance < robot radius)\n")
        print("| corruption | level | " + " | ".join(NAMES[t] for t in TAGS) + " |")
        print("|---|---|" + "---|" * len(TAGS))
        for kind in ("boundary_shift", "thin_obstacle_miss", "boundary_noise"):
            levels = sorted({k[2] for k in a if k[0] == g and k[1] == kind and k[2] >= 0})
            for lv in levels:
                cells = [pct(a[(g, kind, lv, t)][2], a[(g, kind, lv, t)][0]) for t in TAGS]
                print(f"| {kind} | {lv:g} | " + " | ".join(cells) + " |")
        print()

    print("## Stranding reasons (all planners, all levels)\n")
    for kind, rs in strand_reason.items():
        print(f"- {kind}: " + ", ".join(f"{k} {v}" for k, v in sorted(rs.items())))
    print()

    print("## Part B: per-frame rates, pooled over planners, raw → filtered\n")
    print("| bad frame model | p_bad | collision % | margin violation % | stranded % |")
    print("|---|---|---|---|---|")
    for bad in sorted({k[0] for k in b}):
        for p in sorted({k[1] for k in b if k[0] == bad}):
            raw, fil = b[(bad, p, "raw")], b[(bad, p, "filtered")]
            cells = [f"{pct(raw[i], raw[0])} → {pct(fil[i], fil[0])}" for i in (1, 2, 3)]
            print(f"| {bad} | {p:g} | " + " | ".join(cells) + " |")
    print()

    print("## Part B: collision % per planner at p_bad = 0.2, raw → filtered\n")
    print("| bad frame model | " + " | ".join(NAMES[t] for t in TAGS) + " |")
    print("|---|" + "---|" * len(TAGS))
    for bad in sorted({k[0] for k in b}):
        cells = []
        for t in TAGS:
            raw, fil = b_tag[(bad, 0.2, "raw", t)], b_tag[(bad, 0.2, "filtered", t)]
            cells.append(f"{pct(raw[1], raw[0])} → {pct(fil[1], fil[0])}")
        print(f"| {bad} | " + " | ".join(cells) + " |")
    print()

    print("## Part B: outlier-gate detection (all streams with p_bad > 0)\n")
    print("| bad frame model | bad frames | rejected | precision | recall |")
    print("|---|---|---|---|---|")
    for bad, (n_bad, n_rej, n_rej_bad) in sorted(det.items()):
        prec = "–" if n_rej == 0 else f"{100 * n_rej_bad / n_rej:.0f}%"
        rec = "–" if n_bad == 0 else f"{100 * n_rej_bad / n_bad:.0f}%"
        print(f"| {bad} | {n_bad} | {n_rej} | {prec} | {rec} |")


if __name__ == "__main__":
    main()
