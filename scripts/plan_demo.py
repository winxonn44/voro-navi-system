"""Render the three planners on example maps (light: two small maps, one
query each, a few seconds of CPU).

    python scripts/plan_demo.py --out docs/q2

Writes q2_example.png: occupancy, the GVD roadmap, and the Voronoi, A* and
RRT* paths, with each path's length and clearance in the legend. Metrics
use the exact EDT, independent of the JFA field the roadmap was built on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voronav.map_sources import two_rooms
from voronav.maps import random_obstacles
from voronav.planning import (
    astar_plan,
    build_roadmap,
    exact_clearance,
    path_metrics,
    rrt_star_plan,
    voronoi_plan,
)

# Validated categorical palette (dataviz reference, light mode), fixed order.
PLANNERS = {
    "voronoi": ("#2a78d6", "Voronoi roadmap"),
    "astar": ("#eb6834", "Grid A*"),
    "rrt_star": ("#1baf7a", "RRT* (baseline)"),
}
SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#8a897f"
OBSTACLE = "#3d3d3a"
ROADMAP = "#c9c8c0"


def panel(ax, occ, start, goal, title, robot_radius=2.0, seed=0):
    rm = build_roadmap(occ, robot_radius=robot_radius)
    clr = exact_clearance(occ)
    img = np.full(occ.shape + (3,), matplotlib.colors.to_rgb(SURFACE))
    img[rm.skeleton] = matplotlib.colors.to_rgb(ROADMAP)
    img[occ] = matplotlib.colors.to_rgb(OBSTACLE)
    ax.imshow(img, interpolation="nearest")
    results = [
        voronoi_plan(rm, start, goal),
        astar_plan(rm.safe, start, goal),
        rrt_star_plan(rm.safe, start, goal, iters=2000, seed=seed),
    ]
    for r in results:
        color, label = PLANNERS[r.planner]
        if not r.success:
            ax.plot([], [], color=color, label=f"{label}: failed")
            continue
        m = path_metrics(r.path, clr, rm.safe)
        ax.plot(r.path[:, 1], r.path[:, 0], color=color, linewidth=2,
                solid_capstyle="round",
                label=f"{label}: {m['length_px']:.0f} px long, "
                      f"min/mean clearance {m['min_clearance_px']:.1f}/{m['mean_clearance_px']:.1f}")
    for (y, x), name in ((start, "start"), (goal, "goal")):
        ax.plot(x, y, "o", markersize=9, color=INK, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(name, (x, y), xytext=(7, -7), textcoords="offset points", color=INK, fontsize=9)
    ax.plot([], [], color=ROADMAP, linewidth=4, label=f"GVD roadmap ({rm.n_nodes} px, built in {rm.build_s * 1e3:.0f} ms)")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5,
              labelcolor=INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/q2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # warm up the JIT so the build times in the legend are steady-state
    panel(plt.subplots()[1], two_rooms(), (12, 12), (106, 186), "warmup")
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    panel(axes[0], random_obstacles(256, 256, 25, seed=2), (240, 20), (20, 235),
          "Random obstacles, 256 x 256", seed=2)
    panel(axes[1], two_rooms(), (12, 12), (106, 186), "Two rooms and a door, 120 x 200")
    fig.tight_layout()
    fig.savefig(args.out / "q2_example.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out / 'q2_example.png'}")


if __name__ == "__main__":
    main()
