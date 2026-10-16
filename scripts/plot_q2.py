"""Plot the Q2 study from a bench_plan.py results.jsonl.

    python scripts/plot_q2.py runs/bench_q2/results.jsonl --out docs/q2

Writes five figures, plus one tradeoff figure per smoothing mode in the file:
  q2_tradeoff.png    mean / minimum clearance vs path length relative to A*
                     on the same query; the weighted-A* lambda sweep is a curve
  q2_tradeoff_<mode>.png  the same for paths after the <mode> smoothing,
                     relative to the smoothed A* path
  q2_success.png     success rate per planner
  q2_query_time.png  query-time distribution per planner (log scale)
  q2_build_time.png  roadmap build time vs map size
  q2_amortized.png   build + n queries vs n, Voronoi roadmap vs grid A*
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

# Validated categorical palette (dataviz reference, light mode), same fixed
# order as plot_q1.py and plan_demo.py: color follows the planner.
PLANNERS = {
    "voronoi": ("#2a78d6", "Voronoi roadmap"),
    "astar": ("#eb6834", "Grid A*"),
    "rrt_star": ("#1baf7a", "RRT*"),
    "weighted_astar": ("#eda100", "Clearance-weighted A*"),
}
# reads as a progression: shortest (A*) -> lambda sweep -> max clearance (Voronoi)
ORDER = ("astar", "weighted_astar", "voronoi", "rrt_star")
MARKERS = {"random_obstacles": "o", "corridor": "s", "two_rooms": "D",
           "movingai_dir": "^", "image_dir": "v"}
SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5e0"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plain_log_labels(ax, axis: str = "both") -> None:
    """Label log axes 0.1, 1, 10 instead of 10^-1 ... and 2x10^1."""
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        a.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))  # labels even within a decade
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def load(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["row"] == "map"], [r for r in rows if r["row"] == "query"]


Key = tuple[str, float | None, str | None]  # planner, lambda, Voronoi variant
# Voronoi variants share the planner's color; marker shapes tell them apart
VARIANT_MARKERS = ("o", "s", "D", "^", "v", "P")


def series_key(r: dict) -> Key:
    return r["planner"], r["lam"], r.get("variant")


def series_label(key: Key) -> str:
    planner, lam, variant = key
    if lam is not None:
        return "Weighted A*, λ=%g" % lam
    return PLANNERS[planner][1] + (f", {variant}" if variant else "")


def ordered(keys) -> list[Key]:
    variants = sorted({k[2] for k in keys if k[2]})
    return sorted(keys, key=lambda k: (ORDER.index(k[0]), k[1] or 0.0,
                                       variants.index(k[2]) if k[2] else -1))


def smoothing_modes(queries: list[dict]) -> list[str]:
    return sorted({k[: -len("_smooth_s")] for r in queries for k in r if k.endswith("_smooth_s")})


def plot_tradeoff(queries: list[dict], out: Path, mode: str | None = None) -> None:
    p = f"{mode}_" if mode else ""
    astar_len = {(r["map"], r["query"]): r[p + "length_px"] for r in queries
                 if r["planner"] == "astar" and r["success"]}
    pts: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    for r in queries:
        ref = astar_len.get((r["map"], r["query"]))
        if r["success"] and ref:
            pts[series_key(r)].append(
                (r[p + "length_px"] / ref, r[p + "mean_clearance_px"], r[p + "min_clearance_px"]))
    if not pts:
        return
    variants = sorted({k[2] for k in pts if k[2]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax, col, what in ((axes[0], 1, "mean"), (axes[1], 2, "minimum")):
        curve = []
        for key in ordered(pts):
            arr = np.array(pts[key])
            x, y = arr[:, 0].mean(), arr[:, col].mean()
            xe = arr[:, 0].std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
            ye = arr[:, col].std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
            color, _ = PLANNERS[key[0]]
            label = series_label(key)
            if key[0] == "weighted_astar":
                curve.append((x, y))
                ax.errorbar(x, y, xerr=xe, yerr=ye, color=color, marker="o", markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none",
                            elinewidth=1, zorder=3)
                ax.annotate(f"λ={key[1]:g}", (x, y), xytext=(6, -10), textcoords="offset points",
                            color=MUTED, fontsize=8)
                continue
            marker = VARIANT_MARKERS[variants.index(key[2]) % len(VARIANT_MARKERS)] if key[2] else "o"
            ax.errorbar(x, y, xerr=xe, yerr=ye, color=color, marker=marker, markersize=10,
                        markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                        elinewidth=1, zorder=4, label=f"{label} (n={len(arr)})")
        if curve:
            color, label = PLANNERS["weighted_astar"]
            n = len(pts[next(k for k in pts if k[0] == "weighted_astar")])
            ax.plot(*zip(*curve), color=color, linewidth=2, zorder=2,
                    label=f"{label}, λ sweep (n={n})")
        style(ax, f"{what.capitalize()} clearance vs detour"
              + (f", after {mode} smoothing" if mode else ""),
              "path length / A* length on the same query", f"{what} clearance along path (px)")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.text(0.01, 0.005, "Means over successful queries, bars = ±1 standard error. Clearance "
             "from the exact EDT. RRT* runs on a query subset."
             + (f" Every path, A* included, smoothed the same way ({mode})." if mode else ""),
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / (f"q2_tradeoff_{mode}.png" if mode else "q2_tradeoff.png"), dpi=160)
    plt.close(fig)


def plot_success(queries: list[dict], out: Path) -> None:
    by: dict[tuple, list[bool]] = defaultdict(list)
    for r in queries:
        by[series_key(r)].append(bool(r["success"] and r["collision_free"]))
    keys = ordered(by)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 2), 4.5), facecolor=SURFACE)
    rates = [100 * np.mean(by[k]) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, rates, 0.7, color=[PLANNERS[k[0]][0] for k in keys], edgecolor=SURFACE)
    for xi, k, v in zip(x, keys, rates):
        ax.annotate(f"{v:.0f}%\n{sum(by[k])}/{len(by[k])}", (xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [series_label(k).replace(", ", "\n") for k in keys], fontsize=8)
    ax.set_ylim(0, 115)
    style(ax, "Success rate (path found and collision-free)", "", "queries solved (%)")
    fig.tight_layout()
    fig.savefig(out / "q2_success.png", dpi=160)
    plt.close(fig)


def plot_query_time(queries: list[dict], out: Path) -> None:
    by: dict[tuple, list[float]] = defaultdict(list)
    for r in queries:
        by[series_key(r)].append(r["plan_s"] * 1e3)
    keys = ordered(by)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 2), 4.5), facecolor=SURFACE)
    bp = ax.boxplot([by[k] for k in keys], widths=0.6, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, markeredgecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(PLANNERS[k[0]][0])
        patch.set_edgecolor(SURFACE)
    for i, k in enumerate(keys, start=1):
        top = max(bp["caps"][2 * i - 1].get_ydata()[0], max(by[k]))
        ax.annotate(f"median\n{np.median(by[k]):.2g} ms", (i, top), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", color=INK,
                    fontsize=7.5)
    ax.set_xticks(range(1, len(keys) + 1), [series_label(k).replace(", ", "\n") for k in keys],
                  fontsize=8)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4)  # room for the labels above the top whisker
    plain_log_labels(ax, "y")
    style(ax, "Query time (roadmap build excluded)", "", "plan time per query (ms, log)")
    fig.text(0.01, 0.005, "RRT* is pure Python/NumPy; the grid planners are compiled Numba.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "q2_query_time.png", dpi=160)
    plt.close(fig)


def plot_build_time(maps: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    color = PLANNERS["voronoi"][0]
    for source in sorted({m["source"] for m in maps}):
        ms = [m for m in maps if m["source"] == source]
        ax.plot([m["pixels"] for m in ms], [m["build_s"] * 1e3 for m in ms],
                MARKERS.get(source, "o"), color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, linestyle="none", label=source.replace("_", " "))
    px = np.array([m["pixels"] for m in maps], dtype=float)
    bs = np.array([m["build_s"] for m in maps]) * 1e3
    if len(np.unique(px)) >= 2:
        slope, icpt = np.polyfit(np.log(px), np.log(bs), 1)
        xs = np.geomspace(px.min(), px.max(), 50)
        ax.plot(xs, np.exp(icpt) * xs ** slope, color=MUTED, linewidth=1.5, linestyle="--",
                label=f"power-law fit, time ∝ pixels^{slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_labels(ax)
    style(ax, "GVD roadmap build time (JFA + extraction + repair)", "map pixels (log)",
          "build time (ms, log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "q2_build_time.png", dpi=160)
    plt.close(fig)


def plot_amortized(maps: list[dict], queries: list[dict], out: Path) -> None:
    """Uses the first Voronoi variant in the file (the config's first entry)."""
    first = next((r.get("variant") for r in queries if r["planner"] == "voronoi"), None)
    mean_q: dict[tuple[str, str], float] = {}
    for planner in ("voronoi", "astar"):
        by: dict[str, list[float]] = defaultdict(list)
        for r in queries:
            if r["planner"] == planner and (planner != "voronoi" or r.get("variant") == first):
                by[r["map"]].append(r["plan_s"])
        mean_q.update({(planner, k): float(np.mean(v)) for k, v in by.items()})
    names = [m["map"] for m in maps if ("voronoi", m["map"]) in mean_q
             and ("astar", m["map"]) in mean_q]
    if not names:
        return
    build = {m["map"]: m["build_s"] for m in maps}
    n = np.unique(np.geomspace(1, 1e5, 200).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    vor_c, vor_l = PLANNERS["voronoi"]
    ast_c, ast_l = PLANNERS["astar"]
    for name in names:  # per map, faint
        ax.plot(n, build[name] + n * mean_q[("voronoi", name)], color=vor_c, alpha=0.18, lw=1)
        ax.plot(n, n * mean_q[("astar", name)], color=ast_c, alpha=0.18, lw=1)
    b = np.mean([build[k] for k in names])
    tv = np.mean([mean_q[("voronoi", k)] for k in names])
    ta = np.mean([mean_q[("astar", k)] for k in names])
    vor_l += f" ({first})" if first else ""
    ax.plot(n, b + n * tv, color=vor_c, lw=2.5, label=f"{vor_l}: build + n queries")
    ax.plot(n, n * ta, color=ast_c, lw=2.5, label=f"{ast_l}: n queries")
    if ta > tv:
        cross = b / (ta - tv)
        ax.axvline(cross, color=MUTED, lw=1, ls="--")
        ax.annotate(f"break-even ≈ {cross:.0f} queries", (cross, b), xytext=(6, 0),
                    textcoords="offset points", color=INK, fontsize=9)
    else:
        ax.annotate("Voronoi queries are not faster than A* here: no break-even",
                    (0.02, 0.95), xycoords="axes fraction", color=INK, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, f"Amortized time per map (mean of {len(names)} maps; faint = each map)",
          "queries per map n (log)", "total time (s, log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(0, 0.9))
    fig.tight_layout()
    fig.savefig(out / "q2_amortized.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/q2"))
    args = ap.parse_args()
    maps, queries = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(queries, args.out)
    for mode in smoothing_modes(queries):
        plot_tradeoff(queries, args.out, mode)
    plot_success(queries, args.out)
    plot_query_time(queries, args.out)
    plot_build_time(maps, args.out)
    plot_amortized(maps, queries, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
