"""Plot the Q3 study from a bench_q3.py results.jsonl.

    python scripts/plot_q3.py runs/bench_q3/results.jsonl --out docs/q3

Part A: one row per corruption kind, one column per map group ("structural":
every map but cluttered_room; "cluttered": cluttered_room, where any error
that grows the floor also erases 1-2 px chair legs), one line per planner:
  q3_collision.png      share of queries whose path enters a true obstacle
  q3_margin.png         share whose true clearance drops below the robot radius
  q3_stranded.png       share stranded (endpoint unsafe in the corrupted map,
                        or no path found there)
  q3_min_clearance.png  true minimum clearance of the paths that were found
Part B (one column per bad-frame model):
  q3_frames.png         collision (primary), margin-violation and stranded
                        frame rates vs p_bad, raw frames (solid) vs
                        temporal_filter (dashed)
  q3_filter_detection.png  the filter's outlier gate: precision / recall
                        against the injected bad frames, and the IoU it saw
                        on good vs bad frames next to its threshold
Rates are over all queries (a stranded query is neither a collision nor a
violation), bars = ±1 binomial standard error.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_q2 import INK, MUTED, ORDER, PLANNERS, SURFACE, style

KIND_LABELS = {
    "boundary_shift": ("Boundary shift", "shift (px); > 0 = floor grows into walls"),
    "region_dropout": ("Region dropout", "fraction of floor removed"),
    "thin_obstacle_miss": ("Thin obstacles missed", "obstacles thinner than (px) vanish"),
    "boundary_noise": ("Boundary noise", "noise amplitude (px)"),
}
GROUPS = {"structural": "structural", "cluttered": "cluttered_room"}
# within one planner family (same color), later members get these
LINESTYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "D", "^")
GOOD, BAD = "#8a897f", "#d6455d"  # good / injected-bad frames in the detection plot


def load(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["row"]].append(r)
    return out


def group_of(r: dict) -> str:
    return r.get("group") or ("cluttered" if r["source"] == "cluttered_room" else "structural")


def tag_order(rows: list[dict]) -> list[str]:
    seen = {r["tag"]: r for r in rows}
    return sorted(seen, key=lambda t: (ORDER.index(seen[t]["planner"]), seen[t]["lam"] or 0.0,
                                       seen[t]["variant"] or "", t))


def tag_style(rows: list[dict]) -> dict[str, dict]:
    """Color by planner; line style and marker by rank within the planner."""
    first = {r["tag"]: r for r in rows}
    rank: dict[str, int] = defaultdict(int)
    out = {}
    for t in tag_order(rows):
        r = first[t]
        i = rank[r["planner"]]
        rank[r["planner"]] += 1
        name = PLANNERS[r["planner"]][1]
        if r["lam"] is not None:
            name = "Weighted A*, λ=%g" % r["lam"]
        elif r["variant"]:
            name = f"Voronoi, {r['variant']}"
        out[t] = dict(color=PLANNERS[r["planner"]][0], ls=LINESTYLES[i % 4],
                      marker=MARKERS[i % 4], label=name)
    return out


def rate(vals: list[bool]) -> tuple[float, float]:
    p = float(np.mean(vals))
    return 100 * p, 100 * np.sqrt(p * (1 - p) / len(vals)) if len(vals) > 1 else 0.0


def mean_se(vals: list[float]) -> tuple[float, float]:
    return float(np.mean(vals)), (np.std(vals, ddof=1) / np.sqrt(len(vals))
                                  if len(vals) > 1 else 0.0)


def kind_group_grid(queries: list[dict], value, ylabel: str, title: str, note: str,
                    out: Path, ylim=None, hline=None) -> None:
    """Rows = corruption kinds, columns = map groups, one line per planner.
    `value(rows) -> (y, err)` aggregates the rows of one planner and level."""
    kinds = [k for k in KIND_LABELS if any(r["kind"] == k for r in queries)]
    kinds += sorted({r["kind"] for r in queries} - set(kinds))
    groups = [g for g in GROUPS if any(group_of(r) == g for r in queries)]
    styles = tag_style(queries)
    fig, axes = plt.subplots(len(kinds), len(groups), figsize=(6.2 * len(groups),
                             3.6 * len(kinds)), facecolor=SURFACE, squeeze=False)
    for i, kind in enumerate(kinds):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            rows = [r for r in queries if r["kind"] == kind and group_of(r) == group]
            for t in tag_order(rows):
                by: dict[float, list[dict]] = defaultdict(list)
                for r in rows:
                    if r["tag"] == t:
                        by[r["level"]].append(r)
                pts = [(x, *value(by[x])) for x in sorted(by) if by[x]]
                pts = [p for p in pts if p[1] is not None]
                if not pts:
                    continue
                xs, ys, es = zip(*pts)
                st = styles[t]
                ax.errorbar(xs, ys, yerr=es, color=st["color"], linestyle=st["ls"],
                            marker=st["marker"], markersize=5, markeredgecolor=SURFACE,
                            linewidth=1.6, elinewidth=0.8, capsize=0, label=st["label"])
            if hline is not None:
                ax.axhline(hline[0], color=MUTED, linewidth=1, linestyle="--")
                ax.annotate(hline[1], (1, hline[0]), xycoords=("axes fraction", "data"),
                            xytext=(-4, 3), textcoords="offset points", ha="right",
                            color=MUTED, fontsize=8)
            n_q = len({(r["map"], r["query"]) for r in rows})
            n_m = len({r["map"] for r in rows})
            name, xlabel = KIND_LABELS.get(kind, (kind, "level"))
            style(ax, f"{name} · {GROUPS[group]} ({n_m} maps, {n_q} q)", xlabel, ylabel)
            ax.title.set_fontsize(10)
            if ylim:
                ax.set_ylim(*ylim)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.003, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)


GROUP_NOTE = ("Structural = every map but cluttered_room. On cluttered_room an error that "
              "grows the floor also erases 1-2 px chair legs, so the groups are kept apart.\n")


def plot_rate(queries, key, title, ylabel, out, note) -> None:
    kind_group_grid(queries, lambda rows: rate([bool(r[key]) for r in rows]), ylabel, title,
                    GROUP_NOTE + note, out, ylim=(-3, 103))


def plot_min_clearance(queries, radius, out) -> None:
    def value(rows):
        vals = [r["min_clearance_px"] for r in rows if not r["stranded"]]
        return mean_se(vals) if vals else (None, None)

    kind_group_grid(queries, value, "true min clearance (px)",
                    "True minimum clearance of the paths found on the corrupted map",
                    GROUP_NOTE + "Means over non-stranded queries (each point averages a "
                    "different subset), bars = ±1 SE. Exact EDT of the true map.", out,
                    hline=None if radius is None else (radius, f"robot radius {radius:g} px"))


def plot_frames(seqs: list[dict], out: Path) -> None:
    bads = sorted({tuple(r["bad"]) for r in seqs})
    styles = tag_style(seqs)
    metrics = (("collision_frac", "collisions (% of frames)"),
               ("margin_violation_frac", "margin violations (% of frames)"),
               ("stranded_frac", "stranded (% of frames)"))
    fig, axes = plt.subplots(len(metrics), len(bads), figsize=(6.2 * len(bads), 11),
                             facecolor=SURFACE, squeeze=False)
    for col, bad in enumerate(bads):
        rows = [r for r in seqs if tuple(r["bad"]) == bad]
        n_q = len({(r["map"], r["query"]) for r in rows})
        n_m = len({r["map"] for r in rows})
        name = KIND_LABELS.get(bad[0], (bad[0],))[0]
        for row_i, (key, ylabel) in enumerate(metrics):
            ax = axes[row_i, col]
            for t in tag_order(rows):
                st = styles[t]
                for stream, ls in (("raw", "-"), ("filtered", "--")):
                    by: dict[float, list[float]] = defaultdict(list)
                    for r in rows:
                        if r["tag"] == t and r["stream"] == stream:
                            by[r["p_bad"]].append(r[key])
                    xs = sorted(by)
                    ax.plot(xs, [100 * np.mean(by[x]) for x in xs], color=st["color"],
                            linestyle=ls, marker=st["marker"], markersize=4.5,
                            markeredgecolor=SURFACE, linewidth=1.5,
                            label=f"{st['label']}, {stream}")
            style(ax, f"Bad = {name.lower()} {bad[1]:g} ({n_m} maps, {n_q} queries)"
                  if row_i == 0 else "", "probability a frame is bad (p_bad)", ylabel)
            ax.set_ylim(-3, 103)
    axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left", ncol=2)
    base = seqs[0].get("base")
    fig.suptitle("Bad frames: plan on every frame, raw (solid) vs temporal filter (dashed)",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.004, "Collision = path enters a true obstacle (primary). Margin violation "
             "= true clearance below the robot radius. Good frames: "
             + (f"{base[0]} {base[1]:g}." if base else "exact map."), color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_detection(filters: list[dict], outlier_iou: float | None, out: Path) -> None:
    bads = sorted({tuple(r["bad"]) for r in filters})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(bads))
    for k, (what, color, dx) in enumerate((("precision", INK, -0.18), ("recall", BAD, 0.18))):
        vals, labels = [], []
        for bad in bads:
            rs = [r for r in filters if tuple(r["bad"]) == bad]
            tp = sum(r["n_rejected_bad"] for r in rs)
            denom = sum(r["n_rejected"] if what == "precision" else r["n_bad"] for r in rs)
            vals.append(100 * tp / denom if denom else np.nan)
            labels.append(f"{tp}/{denom}")
        ax.bar(x + dx, np.nan_to_num(vals), 0.36, color=color, edgecolor=SURFACE, label=what)
        for xi, v, lab in zip(x + dx, vals, labels):
            ax.annotate("n/a" if np.isnan(v) else f"{v:.0f}%\n{lab}", (xi, np.nan_to_num(v)),
                        xytext=(0, 3), textcoords="offset points", ha="center", color=INK,
                        fontsize=8)
    ax.set_xticks(x, [f"{KIND_LABELS.get(b[0], (b[0],))[0]} {b[1]:g}" for b in bads],
                  fontsize=9)
    ax.set_ylim(0, 118)
    style(ax, "Outlier gate vs injected bad frames", "", "%")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, bad in enumerate(bads):
        for is_bad, color, dx in ((False, GOOD, -0.15), (True, BAD, 0.15)):
            ious = [iou for r in filters if tuple(r["bad"]) == bad
                    for iou, b in zip(r["gate_iou"], r["is_bad"]) if iou is not None and b == is_bad]
            if ious:
                ax.scatter(i + dx + rng.uniform(-0.08, 0.08, len(ious)), ious, s=9,
                           color=color, alpha=0.6, linewidths=0,
                           label=("injected bad" if is_bad else "good") if i == 0 else None)
    if outlier_iou is not None:
        ax.axhline(outlier_iou, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"reject below {outlier_iou:g}", (1, outlier_iou),
                    xycoords=("axes fraction", "data"), xytext=(-4, -12),
                    textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(bads)), [f"{KIND_LABELS.get(b[0], (b[0],))[0]} {b[1]:g}"
                                     for b in bads], fontsize=9)
    style(ax, "Floor IoU with the last accepted mask (what the gate sees)", "", "free IoU")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.text(0.01, 0.005, "Precision = rejected frames that were injected bad; recall = "
             "injected bad frames rejected. A frame let through by the patience reset counts "
             "as not rejected. Pooled over maps and p_bad.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/q3"))
    ap.add_argument("--robot-radius", type=float, default=None,
                    help="default: read from config.json next to the results")
    args = ap.parse_args()
    rows = load(args.results)
    cfg_path = args.results.parent / "config.json"
    cfg = json.loads(cfg_path.read_text())["q3"] if cfg_path.exists() else None
    radius = args.robot_radius if args.robot_radius is not None else (
        cfg["robot_radius"] if cfg else None)
    args.out.mkdir(parents=True, exist_ok=True)
    queries = rows["query"]
    if queries:
        plot_rate(queries, "collision", "Collision rate: path enters a true obstacle",
                  "collisions (% of queries)", args.out / "q3_collision.png",
                  "Plan on the corrupted map, check against the true map. Level 0 = control.")
        plot_rate(queries, "margin_violation",
                  "Margin violations: true clearance below the robot radius",
                  "violations (% of queries)", args.out / "q3_margin.png",
                  "Every collision is also a violation. Level 0 = control.")
        stranded = [r for r in queries if r["stranded"]]
        share = (f" {100 * np.mean([r['strand_reason'] == 'endpoint' for r in stranded]):.0f}% "
                 "of strandings are an unsafe endpoint, the rest no path." if stranded else "")
        plot_rate(queries, "stranded", "Stranded: endpoint unsafe or no path in corrupted map",
                  "stranded (% of queries)", args.out / "q3_stranded.png",
                  "The control strands nothing. All planners are complete on the same "
                  "corrupted safe mask, so they strand together." + share)
        plot_min_clearance(queries, radius, args.out / "q3_min_clearance.png")
    if rows["sequence"]:
        plot_frames(rows["sequence"], args.out / "q3_frames.png")
    if rows["filter"]:
        plot_detection(rows["filter"], cfg["frames"]["outlier_iou"] if cfg else None,
                       args.out / "q3_filter_detection.png")
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
