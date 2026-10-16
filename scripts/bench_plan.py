"""Q2 benchmark harness: Voronoi roadmap vs grid A* vs clearance-weighted A*
vs RRT* on the same seeded queries. One JSON line per map and one per
(query, planner).

    python scripts/bench_plan.py --config configs/bench_q2_quick.yaml  # < 1 min, everyday check
    python scripts/bench_plan.py --config configs/bench_q2.yaml        # full study, run plugged in

Rows have "row": "map" (roadmap stats, machine info) or "row": "query"
(success, plan_s, expansions and every path_metrics field, plus the same
fields prefixed `<mode>_` for every q2.smoothing mode). Path metrics use
the exact SciPy EDT of the map, never the JFA field the roadmap and the
weighted A* cost map are built from. Queries are start/goal pairs in the same
8-connected safe region, at least `min_dist_frac` of the map diagonal apart.
Voronoi runs once per q2.voronoi_variants entry (row field "variant").
RRT* runs on every `rrt_every`-th query only; its rows share query ids with
the other planners, so comparisons can be restricted to the same queries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from bench_jfa import machine_info
from voronav.config import Q2Config, load_q2_config, to_dict
from voronav.map_sources import expand_all, two_rooms
from voronav.planning.grid_search import REACHABILITY
from voronav.planning import (
    astar_plan,
    build_roadmap,
    exact_clearance,
    path_metrics,
    rrt_star_plan,
    smooth_path,
    voronoi_plan,
    weighted_astar_plan,
)

METRIC_KEYS = ("length_px", "min_clearance_px", "mean_clearance_px", "p10_clearance_px",
               "total_turn_deg", "collision_free", "n_waypoints")


def sample_queries(safe: np.ndarray, n: int, min_dist: float, rng: np.random.Generator,
                   max_tries: int | None = None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Up to n (start, goal) pairs: start uniform over safe pixels, goal
    uniform over pixels of the start's reachable safe region (4-connected:
    see grid_search.REACHABILITY) that are at
    least `min_dist` away. Fewer than n when the map has no room for them."""
    regions, _ = label(safe, structure=REACHABILITY)
    free = np.argwhere(safe)
    if len(free) == 0:
        return []
    out = []
    for _ in range(max_tries or 50 * n):
        if len(out) == n:
            break
        s = free[rng.integers(len(free))]
        pix = np.argwhere(regions == regions[s[0], s[1]])
        far = pix[np.hypot(*(pix - s).T) >= min_dist]
        if len(far) == 0:
            continue
        g = far[rng.integers(len(far))]
        out.append(((int(s[0]), int(s[1])), (int(g[0]), int(g[1]))))
    return out


def run_planners(cfg: Q2Config, rm, query_id: int, start, goal):
    """(lam, variant, PlanResult) for every configured planner on one query."""
    q2 = cfg.q2
    for planner in q2.planners:
        if planner == "voronoi":
            for v in q2.voronoi_variants:
                yield None, v.name, voronoi_plan(rm, start, goal, join=v.join,
                                                 join_lam=v.join_lam, roadmap_lam=v.roadmap_lam)
        elif planner == "astar":
            yield None, None, astar_plan(rm.safe, start, goal)
        elif planner == "weighted_astar":
            for lam in q2.lambdas:
                yield lam, None, weighted_astar_plan(rm.safe, rm.clearance, start, goal, lam)
        elif planner == "rrt_star" and query_id % q2.rrt_every == 0:
            yield None, None, rrt_star_plan(rm.safe, start, goal, iters=q2.rrt_iters,
                                            seed=q2.seed + query_id)


def smoothed_metrics(cfg: Q2Config, rm, res, clr) -> dict:
    """`<mode>_<metric>` and `<mode>_smooth_s` for every configured smoothing
    mode: the same post-process on every planner's path, graded like the raw
    path (exact EDT). Null on failure."""
    out = {}
    for mode in cfg.q2.smoothing:
        if not res.success:
            out.update({f"{mode}_{k}": None for k in (*METRIC_KEYS, "smooth_s")})
            continue
        t0 = time.perf_counter()
        sm = smooth_path(res.path, rm.safe, rm.clearance, mode)
        out[f"{mode}_smooth_s"] = time.perf_counter() - t0
        out.update({f"{mode}_{k}": v for k, v in path_metrics(sm, clr, rm.safe).items()})
    return out


def build(cfg: Q2Config, occ: np.ndarray):
    r = cfg.roadmap
    return build_roadmap(occ, robot_radius=cfg.q2.robot_radius, min_angle_deg=r.min_angle_deg,
                         raster_slack_px=r.raster_slack_px, min_piece_px=r.min_piece_px,
                         bridge_lam=r.bridge_lam, backend=r.backend, variant=r.variant)


def warmup(cfg: Q2Config) -> None:
    """Compile and load the Numba kernels so the first map's timings are
    steady-state, not JIT time."""
    rm = build(cfg, two_rooms(60, 100))
    light = replace(cfg, q2=replace(cfg.q2, lambdas=cfg.q2.lambdas[:1], rrt_iters=20))
    for _, _, res in run_planners(light, rm, 0, (10, 10), (50, 90)):
        if res.success:
            for mode in cfg.q2.smoothing:
                smooth_path(res.path, rm.safe, rm.clearance, mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override q2.out_dir")
    args = ap.parse_args()

    cfg = load_q2_config(args.config)
    q2 = cfg.q2
    out_dir = Path(args.out or q2.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    machine = machine_info()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    warmup(cfg)

    with open(out_path, "w") as f:
        for map_i, m in enumerate(expand_all(q2.maps)):
            h, w = m.occ.shape
            rm = build(cfg, m.occ)
            clr = exact_clearance(m.occ)  # independent of the JFA field
            # one query stream per map, independent of map order
            rng = np.random.default_rng([q2.seed, map_i])
            queries = sample_queries(rm.safe, q2.queries_per_map,
                                     q2.min_dist_frac * math.hypot(h, w), rng)
            if len(queries) < q2.queries_per_map:
                print(f"  {m.name}: only {len(queries)}/{q2.queries_per_map} queries fit",
                      file=sys.stderr)
            base = {"started": started, "map": m.name, "source": m.source, **m.params,
                    "height": h, "width": w, "pixels": h * w}
            f.write(json.dumps({
                "row": "map", **base,
                "free_frac": float((~m.occ).mean()),
                "safe_frac": float(rm.safe.mean()),
                "robot_radius": rm.robot_radius,
                "build_s": rm.build_s,
                "n_nodes": rm.n_nodes,
                "n_bridges": rm.n_bridges,
                "bridge_px": rm.bridge_px,
                "n_pruned": rm.n_pruned,
                "n_queries": len(queries),
                "roadmap": to_dict(cfg.roadmap),
                "machine": machine,
            }) + "\n")

            summary: dict[str, list] = {}
            for qi, (start, goal) in enumerate(queries):
                for lam, variant, res in run_planners(cfg, rm, qi, start, goal):
                    metrics = (path_metrics(res.path, clr, rm.safe) if res.success
                               else dict.fromkeys(METRIC_KEYS))
                    metrics.update(smoothed_metrics(cfg, rm, res, clr))
                    row = {
                        "row": "query", **base,
                        "query": qi,
                        "start": list(start),
                        "goal": list(goal),
                        "euclid_px": math.dist(start, goal),
                        "planner": res.planner,
                        "lam": lam,
                        "variant": variant,
                        "success": res.success,
                        "plan_s": res.plan_s,
                        "expansions": res.expansions,
                        "fail": res.info.get("fail"),
                        **metrics,
                    }
                    f.write(json.dumps(row) + "\n")
                    tag = res.planner + ("" if lam is None else f"@{lam:g}") + (
                        "" if variant is None else f"[{variant}]")
                    summary.setdefault(tag, []).append(row)
            f.flush()
            parts = []
            for tag, rows in summary.items():
                ok = [r for r in rows if r["success"]]
                clr_mean = np.mean([r["mean_clearance_px"] for r in ok]) if ok else float("nan")
                parts.append(f"{tag} {len(ok)}/{len(rows)} "
                             f"{np.median([r['plan_s'] for r in rows]) * 1e3:.1f}ms c{clr_mean:.1f}")
            print(f"{m.name:28s} {h}x{w} build={rm.build_s * 1e3:6.1f} ms  " + "  ".join(parts))
            if q2.cooldown_s:
                time.sleep(q2.cooldown_s)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
