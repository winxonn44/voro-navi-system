"""Summarize a bench_plan.py results.jsonl into the Markdown tables used in
the README's Q2 section.

    python scripts/summarize_q2.py runs/bench_q2/results.jsonl > docs/q2/q2_summary.md

Every comparison is paired: ratios are computed per query against the other
planner on the *same* start/goal, then aggregated (median for ratios and
times, mean for clearances). RRT* rows exist only on every k-th query, so
its row is computed on that subset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def family(map_name: str) -> str:
    if map_name.startswith("movingai_dir-"):
        part = map_name.split("-")[1]
        # names carry the set folder when the source dir is the MovingAI root
        # (movingai_dir-maze-...); with a per-set dir they do not, so infer it
        for known in ("bg512", "dao", "maze", "room", "street"):
            if part == known or known in part:
                return "MovingAI " + known
        return "MovingAI " + part
    if map_name.startswith("houseexpo_dir-"):
        return "HouseExpo"
    if map_name.startswith("random_obstacles-"):
        return f"random {map_name.split('-')[1]}²"
    return "corridor / two rooms"


def tag(r: dict) -> str:
    if r["planner"] == "voronoi":
        return f"Voronoi [{r['variant']}]"
    if r["planner"] == "weighted_astar":
        return f"weighted A* λ={r['lam']:g}"
    return {"astar": "A*", "rrt_star": "RRT*"}[r["planner"]]


def load(path: Path):
    maps, queries = {}, []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["row"] == "map":
            maps[r["map"]] = r
        else:
            r["family"] = family(r["map"])
            r["tag"] = tag(r)
            queries.append(r)
    return maps, queries


def index(queries):
    """(map, query) -> tag -> row"""
    by = defaultdict(dict)
    for r in queries:
        by[(r["map"], r["query"])][r["tag"]] = r
    return by


ORDER = ["A*", "weighted A* λ=4", "weighted A* λ=16", "weighted A* λ=64", "weighted A* λ=128",
         "Voronoi [nearest]", "Voronoi [nearest-road16]", "Voronoi [best16]", "Voronoi [best64]",
         "RRT*"]


def table(by, tags, prefix=""):
    """One row per planner tag: solved, length/A*, clearances, query time.
    prefix "" = raw paths, "preserve_" = smoothed (min clearance kept)."""
    out = ["| planner | solved | length / A* (median) | mean clearance | min clearance | median query |",
           "|---|---|---|---|---|---|"]
    for t in tags:
        rows = [(q[t], q.get("A*")) for q in by.values() if t in q]
        if not rows:
            continue
        ok = [(r, a) for r, a in rows if r["success"] and a is not None and a["success"]]
        ratio = [r[f"{prefix}length_px"] / a[f"{prefix}length_px"] for r, a in ok
                 if a[f"{prefix}length_px"] > 0]
        out.append(
            f"| {t} | {sum(r['success'] for r, _ in rows)}/{len(rows)} "
            f"| {np.median(ratio):.3f} "
            f"| {np.mean([r[f'{prefix}mean_clearance_px'] for r, _ in ok]):.1f} px "
            f"| {np.mean([r[f'{prefix}min_clearance_px'] for r, _ in ok]):.2f} px "
            f"| {np.median([r['plan_s'] for r, _ in rows]) * 1e3:.2f} ms |"
        )
    return "\n".join(out)


def head_to_head(by, a_tag, b_tag, families):
    """Paired a vs b per family: clearance wins, length and speed ratios."""
    out = [f"| map family | queries | {a_tag} min clearance ≥ {b_tag} | mean clearance a − b "
           f"| length a / b (median) | speedup b_time / a_time (median) |",
           "|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for q in by.values():
        if a_tag in q and b_tag in q and q[a_tag]["success"] and q[b_tag]["success"]:
            groups[q[a_tag]["family"]].append((q[a_tag], q[b_tag]))
    for fam in families + ["all"]:
        pairs = [p for g in groups.values() for p in g] if fam == "all" else groups.get(fam, [])
        if not pairs:
            continue
        ge = np.mean([a["min_clearance_px"] >= b["min_clearance_px"] - 1e-9 for a, b in pairs])
        dmean = np.mean([a["mean_clearance_px"] - b["mean_clearance_px"] for a, b in pairs])
        lr = np.median([a["length_px"] / b["length_px"] for a, b in pairs])
        sp = np.median([b["plan_s"] / max(a["plan_s"], 1e-9) for a, b in pairs])
        name = f"**{fam}**" if fam == "all" else fam
        out.append(f"| {name} | {len(pairs)} | {ge:.0%} | {dmean:+.1f} px | {lr:.3f} | {sp:.1f}× |")
    return "\n".join(out)


def roadmap_table(maps, families):
    out = ["| map family | maps | median size | median build | skeleton px (median) | bridges / map | pruned px / map |",
           "|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for m in maps.values():
        groups[family(m["map"])].append(m)
    for fam in families:
        ms = groups.get(fam, [])
        if not ms:
            continue
        out.append(
            f"| {fam} | {len(ms)} | {int(np.median([max(m['height'], m['width']) for m in ms]))} px "
            f"| {np.median([m['build_s'] for m in ms]) * 1e3:.1f} ms "
            f"| {int(np.median([m['n_nodes'] for m in ms]))} "
            f"| {np.mean([m['n_bridges'] for m in ms]):.1f} "
            f"| {np.mean([m['n_pruned'] for m in ms]):.0f} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    maps, queries = load(args.results)
    by = index(queries)
    tags = [t for t in ORDER if any(t in q for q in by.values())]
    families = sorted({r["family"] for r in queries},
                      key=lambda f: (not f.startswith("random"), f.startswith("Moving"),
                                     f.startswith("House"), f))
    bad = [r for r in queries if r["success"] and not r["collision_free"]]
    n_maps, n_q = len(maps), len(by)
    print(f"# Q2 summary\n\n{n_maps} maps, {n_q} queries, {len(queries)} planner runs. "
          f"Successful paths that failed the independent collision check: {len(bad)}.\n")
    print("## All maps, raw paths\n")
    print(table(by, tags))
    print("\n## All maps, smoothed (`preserve`: minimum clearance never lowered)\n")
    print(table(by, tags, prefix="preserve_"))
    for a, b in (("Voronoi [best64]", "weighted A* λ=64"),
                 ("Voronoi [nearest-road16]", "weighted A* λ=16"),
                 ("Voronoi [nearest]", "A*")):
        if a in tags and b in tags:
            print(f"\n## {a} vs {b}, paired by query\n")
            print(head_to_head(by, a, b, families))
    print("\n## Roadmap build per map family\n")
    print(roadmap_table(maps, families))


if __name__ == "__main__":
    main()
