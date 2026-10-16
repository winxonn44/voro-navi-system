"""Plot the Q1 study from a bench_jfa.py results.jsonl.

    python scripts/plot_q1.py runs/bench_q1/results.jsonl --out docs/q1

Writes three figures:
  q1_throughput.png  time vs grid size, one line per backend (best config)
  q1_scaling.png     Numba speedup vs threads, with an Amdahl's-law fit
  q1_error.png       pixel error rate per JFA variant and site density
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
from scipy.optimize import curve_fit

# Validated categorical palette (dataviz reference, light mode), fixed order:
# color follows the backend, never its rank.
SERIES = {
    "numpy-cpu": ("#2a78d6", "NumPy (reference)"),
    "numba-cpu": ("#eb6834", "Numba, multithreaded"),
    "torch-cpu": ("#1baf7a", "PyTorch CPU"),
    "torch-mps": ("#eda100", "PyTorch MPS (GPU)"),
    "torch-cuda": ("#e87ba4", "PyTorch CUDA (GPU)"),
}
SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5e0"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_throughput(rows: list[dict], out: Path, variant: str, density: float) -> None:
    best: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["variant"] != variant or r["density"] != density:
            continue
        key = f"{r['backend']}-{r['device']}"
        cur = best[key].get(r["size"])
        best[key][r["size"]] = r["median_s"] if cur is None else min(cur, r["median_s"])

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for key, (color, label) in SERIES.items():
        if key not in best:
            continue
        sizes = sorted(best[key])
        ms = [best[key][s] * 1e3 for s in sizes]
        ax.plot(sizes, ms, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(label.split(",")[0].split(" (")[0], (sizes[-1], ms[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style(ax, f"JFA time per diagram ({variant}, {density:g} sites/pixel)",
          "grid side (pixels)", "median time (ms, log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "q1_throughput.png", dpi=160)
    plt.close(fig)


def amdahl(n, p):
    return 1.0 / ((1.0 - p) + p / n)


def plot_scaling(rows: list[dict], out: Path, variant: str, density: float) -> None:
    numba = [r for r in rows if r["backend"] == "numba" and r["variant"] == variant
             and r["density"] == density]
    if not numba:
        return
    size = max(r["size"] for r in numba)
    pts = sorted((r["threads"], r["median_s"]) for r in numba if r["size"] == size)
    base = dict(pts).get(1)
    if base is None or len(pts) < 2:
        return
    threads = np.array([t for t, _ in pts], dtype=float)
    speedup = np.array([base / s for _, s in pts])
    (p,), _ = curve_fit(amdahl, threads, speedup, p0=[0.9], bounds=(0.0, 1.0))

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    xs = np.linspace(1, threads.max(), 100)
    ax.plot(xs, xs, color=MUTED, linewidth=1, linestyle="--", label="ideal (linear)")
    ax.plot(xs, amdahl(xs, p), color=MUTED, linewidth=2,
            label=f"Amdahl fit, parallel fraction p = {p:.3f}")
    color, label = SERIES["numba-cpu"]
    ax.plot(threads, speedup, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=f"{label} (measured)")
    ax.annotate(f"{speedup[-1]:.1f}x", (threads[-1], speedup[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
    style(ax, f"Numba strong scaling, {size}x{size} grid ({variant})",
          "threads", "speedup vs 1 thread")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "q1_scaling.png", dpi=160)
    plt.close(fig)


def plot_error(rows: list[dict], out: Path) -> None:
    if not any("error_rate" in r for r in rows):
        return
    size = max(r["size"] for r in rows)
    err: dict[tuple[str, float], float] = {}
    for r in rows:
        if r["size"] == size and "error_rate" in r:
            err[(r["variant"], r["density"])] = r["error_rate"]
    variants = [v for v in ("jfa", "1+jfa", "jfa+1", "jfa+2") if any(k[0] == v for k in err)]
    densities = sorted({k[1] for k in err})
    # one sequential hue, light -> dark with density (magnitude, not identity)
    ramp = ["#9ec5f4", "#5598e7", "#256abf", "#104281"][-len(densities):] if len(densities) <= 4 \
        else plt.cm.Blues(np.linspace(0.35, 0.95, len(densities)))

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    width = 0.8 / len(densities)
    x = np.arange(len(variants))
    for i, d in enumerate(densities):
        vals = [err.get((v, d), np.nan) * 100 for v in variants]
        xs = x + (i - (len(densities) - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92,
               color=ramp[i], label=f"{d:g} sites/pixel", edgecolor=SURFACE, linewidth=1)
        for xi, v in zip(xs, vals):
            if v == 0:  # an exact zero is a result, not missing data: say so
                ax.annotate("0", (xi, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", color=MUTED, fontsize=8)
    ax.set_xticks(x, variants)
    style(ax, f"JFA pixels with the wrong distance, {size}x{size} grid",
          "variant", "error rate (% of pixels)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="site density",
              title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "q1_error.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/q1"))
    ap.add_argument("--variant", default="jfa+1")
    ap.add_argument("--density", type=float, help="density for time/scaling plots (default: middle)")
    args = ap.parse_args()

    rows = load(args.results)
    densities = sorted({r["density"] for r in rows})
    density = args.density if args.density is not None else densities[len(densities) // 2]
    args.out.mkdir(parents=True, exist_ok=True)
    plot_throughput(rows, args.out, args.variant, density)
    plot_scaling(rows, args.out, args.variant, density)
    plot_error(rows, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
