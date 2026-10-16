"""Q3 benchmark harness: how does segmentation error carry through to path
safety? Corrupt the true map, plan on the corrupted map, grade the path on
the TRUE map.

    python scripts/bench_q3.py --config configs/bench_q3_quick.yaml  # ~30 s, everyday check
    python scripts/bench_q3.py --config configs/bench_q3.yaml        # the study, run plugged in

Part A (static errors): for every map, corruption kind, level and draw
(voronav.corruption.CORRUPTIONS), build the roadmap / safe mask on the
corrupted map and run every planner on the same seeded queries. Queries are
sampled in the TRUE safe space (free, exact clearance >= robot radius, one
search region; see bench_plan.sample_queries), so on the control level every
planner should succeed safely.

Part B (bad frames, q3.frames): a stream of frames where each one is badly
segmented with probability p_bad (temporal.bad_frame_sequence); plan on
every frame, raw and after temporal.temporal_filter.

Rows ("row" field):
  map       one per true map (Part A and Part B maps)
  corrupt   one per corrupted map: kind, level, draw, mask_error_stats,
            roadmap build stats
  query     Part A, one per (corrupted map, query, planner)
  frame     Part B, one per (sequence, stream, frame, query, planner)
  sequence  Part B, one per (sequence, stream, query, planner): fractions of
            frames that were unsafe / collided / stranded
  filter    Part B, one per sequence: what temporal_filter's outlier gate did
            on each frame (accept / reject / reset, the IoU it saw, how many
            injected-bad frames were in the vote) and its detection
            precision / recall against the injected bad frames
Every row carries "group": "cluttered" (cluttered_room) or "structural"
(every other map); analyze them apart (see map_group).

Outcomes, graded on the true map (exact SciPy EDT, never the JFA field):
  stranded          an endpoint is not safe in the corrupted map, or no path
                    was found there (strand_reason says which)
  collision         the path enters a true obstacle pixel (true min clearance 0)
  margin_violation  true clearance < robot radius somewhere (implies nothing
                    about collision; every collision is also a violation)
Stranded rows have collision = margin_violation = False and null metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench_jfa import machine_info
from bench_plan import METRIC_KEYS, sample_queries
from voronav import corruption, temporal
from voronav.config import Q3Config, load_q3_config, to_dict
from voronav.map_sources import expand_all, two_rooms
from voronav.planning import (
    astar_plan,
    build_roadmap,
    exact_clearance,
    path_metrics,
    voronoi_plan,
    weighted_astar_plan,
)

# path_metrics' collision_free is "inside the true safe mask" here
GRADE_KEYS = tuple("in_true_safe" if k == "collision_free" else k for k in METRIC_KEYS)
STATS_KEYS = ("free_iou", "false_free_frac", "false_obstacle_frac", "boundary_error_px")


def planner_specs(cfg: Q3Config) -> list[tuple[str, float | None, object]]:
    """(planner, lambda, Voronoi variant config) for every configured planner."""
    q3 = cfg.q3
    specs = []
    for planner in q3.planners:
        if planner == "voronoi":
            specs += [("voronoi", None, v) for v in q3.voronoi_variants]
        elif planner == "astar":
            specs.append(("astar", None, None))
        elif planner == "weighted_astar":
            specs += [("weighted_astar", lam, None) for lam in q3.lambdas]
    return specs


def tag(spec) -> str:
    planner, lam, v = spec
    return planner + ("" if lam is None else f"@{lam:g}") + ("" if v is None else f"[{v.name}]")


def plan(spec, rm, start, goal):
    planner, lam, v = spec
    if planner == "voronoi":
        return voronoi_plan(rm, start, goal, join=v.join, join_lam=v.join_lam,
                            roadmap_lam=v.roadmap_lam)
    if planner == "astar":
        return astar_plan(rm.safe, start, goal)
    return weighted_astar_plan(rm.safe, rm.clearance, start, goal, lam)


def outcome(spec, rm, start, goal, true_clr, true_safe, radius) -> dict:
    """Plan on the corrupted roadmap `rm`, grade on the true map."""
    stranded = {"stranded": True, "collision": False, "margin_violation": False,
                "plan_s": None, "expansions": None, **dict.fromkeys(GRADE_KEYS)}
    if not (rm.safe[start] and rm.safe[goal]):
        return {**stranded, "strand_reason": "endpoint"}
    res = plan(spec, rm, start, goal)
    if not res.success:
        return {**stranded, "strand_reason": "no_path", "fail": res.info.get("fail"),
                "plan_s": res.plan_s, "expansions": res.expansions}
    m = path_metrics(res.path, true_clr, true_safe)
    m["in_true_safe"] = m.pop("collision_free")
    return {
        "stranded": False,
        "strand_reason": None,
        "collision": m["min_clearance_px"] <= 0.0,
        "margin_violation": m["min_clearance_px"] < radius,
        "plan_s": res.plan_s,
        "expansions": res.expansions,
        **m,
    }


def build(cfg: Q3Config, occ: np.ndarray):
    r = cfg.roadmap
    return build_roadmap(occ, robot_radius=cfg.q3.robot_radius, min_angle_deg=r.min_angle_deg,
                         raster_slack_px=r.raster_slack_px, min_piece_px=r.min_piece_px,
                         bridge_lam=r.bridge_lam, backend=r.backend, variant=r.variant)


def warmup(cfg: Q3Config) -> None:
    """Compile and load the Numba kernels so the first map's timings are
    steady-state, not JIT time."""
    rm = build(cfg, two_rooms(60, 100))
    for spec in planner_specs(cfg):
        plan(spec, rm, (10, 10), (50, 90))


def stable_seed(*parts) -> int:
    """A seed from names and levels that does not depend on config order."""
    return zlib.crc32(":".join(f"{p:g}" if isinstance(p, float) else str(p)
                               for p in parts).encode())


def truth(cfg: Q3Config, occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact clearance and the safe mask the grader uses."""
    clr = exact_clearance(occ)
    return clr, ~occ & (clr >= cfg.q3.robot_radius)


def map_row(m, base, true_safe, queries, part: str, machine) -> dict:
    return {"row": "map", **base, "part": part, "free_frac": float((~m.occ).mean()),
            "safe_frac": float(true_safe.mean()), "n_queries": len(queries),
            "machine": machine}


def map_queries(cfg: Q3Config, m, true_safe, rng, n: int):
    h, w = m.occ.shape
    queries = sample_queries(true_safe, n, cfg.q3.min_dist_frac * math.hypot(h, w), rng)
    if len(queries) < n:
        print(f"  {m.name}: only {len(queries)}/{n} queries fit", file=sys.stderr)
    return queries


def part_a(cfg: Q3Config, f, started: str, machine) -> None:
    q3 = cfg.q3
    specs = planner_specs(cfg)
    for map_i, m in enumerate(expand_all(q3.maps)):
        t_map = time.perf_counter()
        h, w = m.occ.shape
        true_clr, true_safe = truth(cfg, m.occ)
        # one query stream per map, independent of map order
        queries = map_queries(cfg, m, true_safe, np.random.default_rng([q3.seed, map_i]),
                              q3.queries_per_map)
        base = {"started": started, "map": m.name, "source": m.source,
                "group": map_group(m.source), **m.params,
                "height": h, "width": w, "pixels": h * w}
        f.write(json.dumps(map_row(m, base, true_safe, queries, "A", machine)) + "\n")
        summary = []
        for kind, levels in q3.corruptions.items():
            fn = corruption.CORRUPTIONS[kind]
            for level in levels:
                level = float(level)
                for draw in range(q3.draws):
                    rng = np.random.default_rng([q3.seed, map_i, stable_seed(kind, level, draw)])
                    est = fn(m.occ, level, rng)
                    stats = corruption.mask_error_stats(m.occ, est)
                    rm = build(cfg, est)
                    cbase = {**base, "kind": kind, "level": level, "draw": draw,
                             **{k: stats[k] for k in STATS_KEYS}}
                    f.write(json.dumps({
                        "row": "corrupt", **cbase, **stats,
                        "identical": bool(np.array_equal(est, m.occ)),
                        "est_safe_frac": float(rm.safe.mean()),
                        "build_s": rm.build_s, "n_nodes": rm.n_nodes,
                        "n_bridges": rm.n_bridges,
                    }) + "\n")
                    n_unsafe = n_strand = n = 0
                    for qi, (start, goal) in enumerate(queries):
                        for spec in specs:
                            out = outcome(spec, rm, start, goal, true_clr, true_safe,
                                          q3.robot_radius)
                            f.write(json.dumps({
                                "row": "query", **cbase, "query": qi,
                                "start": list(start), "goal": list(goal),
                                "euclid_px": math.dist(start, goal),
                                "planner": spec[0], "lam": spec[1],
                                "variant": None if spec[2] is None else spec[2].name,
                                "tag": tag(spec), **out,
                            }) + "\n")
                            n += 1
                            n_unsafe += out["margin_violation"]
                            n_strand += out["stranded"]
                    summary.append(f"{kind}@{level:g} u{n_unsafe}/s{n_strand}/{n}")
        f.flush()
        print(f"{m.name:34s} {h}x{w} {time.perf_counter() - t_map:6.1f} s  "
              + " ".join(summary))
        if q3.cooldown_s:
            time.sleep(q3.cooldown_s)


def map_group(source: str) -> str:
    """cluttered_room is analyzed apart from the rest: on it, any error that
    grows the floor by a pixel or two (boundary_shift, boundary_noise) also
    erases 1-2 px chair legs, so pooling would mix two failure modes."""
    return "cluttered" if source == "cluttered_room" else "structural"


def filter_trace(frames: list[np.ndarray], window: int, outlier_iou: float,
                 patience: int | None) -> tuple[list[np.ndarray], list[dict]]:
    """temporal.temporal_filter, step for step, plus what its gate decided
    per frame: "first" (always accepted), "accept", "reject" (outlier, the
    last accepted mask is kept) or "reset" (rejected `patience` times in a
    row, so accepted as a new start). Also the IoU the gate saw and the
    frame indices in the vote. part_b checks the masks match the real
    filter's, so the trace cannot silently drift from it."""
    hist: list[int] = []
    rejected = 0
    out, trace = [], []
    for t, f in enumerate(frames):
        iou = None
        if not hist:
            hist.append(t)
            gate = "first"
        else:
            iou = temporal.free_iou(f, frames[hist[-1]])
            if iou >= outlier_iou:
                hist.append(t)
                rejected = 0
                gate = "accept"
            else:
                rejected += 1
                gate = "reject"
                if patience is not None and rejected >= patience:
                    hist = [t]
                    rejected = 0
                    gate = "reset"
        hist = hist[-window:]
        votes = np.sum([frames[i] for i in hist], axis=0)
        out.append(2 * votes >= len(hist))
        trace.append({"gate": gate, "gate_iou": iou, "history": list(hist)})
    return out, trace


def detection(is_bad: np.ndarray, trace: list[dict]) -> dict:
    """How well the gate's rejections pick out the injected bad frames."""
    rej = np.array([t["gate"] == "reject" for t in trace])
    tp = int((rej & is_bad).sum())
    return {"n_rejected": int(rej.sum()), "n_rejected_bad": tp,
            "precision": tp / rej.sum() if rej.any() else None,
            "recall": tp / is_bad.sum() if is_bad.any() else None}


FRAME_KEYS = ("stranded", "strand_reason", "collision", "margin_violation",
              "min_clearance_px", "length_px")


def part_b(cfg: Q3Config, f, started: str, machine) -> None:
    q3, fr = cfg.q3, cfg.q3.frames
    specs = planner_specs(cfg)
    for map_i, m in enumerate(expand_all(fr.maps)):
        t_map = time.perf_counter()
        h, w = m.occ.shape
        true_clr, true_safe = truth(cfg, m.occ)
        queries = map_queries(cfg, m, true_safe, np.random.default_rng([q3.seed, map_i, 1]),
                              fr.queries_per_map)
        base = {"started": started, "map": m.name, "source": m.source,
                "group": map_group(m.source), **m.params,
                "height": h, "width": w, "pixels": h * w}
        f.write(json.dumps(map_row(m, base, true_safe, queries, "B", machine)) + "\n")
        summary = []
        for bad in fr.bad:
            for p_bad in fr.p_bad:
                rng = np.random.default_rng([q3.seed, map_i, 2, stable_seed(*bad, p_bad)])
                raw, is_bad = temporal.bad_frame_sequence(
                    m.occ, fr.n_frames, p_bad, rng, base=fr.base, bad=bad, return_flags=True)
                filt = temporal.temporal_filter(raw, window=fr.window,
                                                outlier_iou=fr.outlier_iou,
                                                patience=fr.patience)
                mirror, trace = filter_trace(raw, fr.window, fr.outlier_iou, fr.patience)
                if not all(np.array_equal(a, b) for a, b in zip(filt, mirror)):
                    raise RuntimeError("filter_trace no longer mirrors temporal.temporal_filter")
                n_vote_bad = [int(is_bad[tr["history"]].sum()) for tr in trace]
                sbase = {**base, "base": list(fr.base) if fr.base else None,
                         "bad": list(bad), "p_bad": p_bad,
                         "n_frames": fr.n_frames, "n_bad": int(np.sum(is_bad))}
                f.write(json.dumps({
                    "row": "filter", **sbase, **detection(is_bad, trace),
                    "is_bad": [bool(b) for b in is_bad],
                    "gate": [tr["gate"] for tr in trace],
                    "gate_iou": [tr["gate_iou"] for tr in trace],
                    "n_vote": [len(tr["history"]) for tr in trace],
                    "n_vote_bad": n_vote_bad,
                }) + "\n")
                for stream, seq in (("raw", raw), ("filtered", filt)):
                    cache: dict[bytes, tuple] = {}  # identical frames: build once
                    tally: dict[tuple[int, str], list[dict]] = {}
                    for t, est in enumerate(seq):
                        key = hashlib.blake2b(est.tobytes(), digest_size=16).digest()
                        if key not in cache:
                            cache[key] = (build(cfg, est), corruption.mask_error_stats(m.occ, est))
                        rm, stats = cache[key]
                        tr = trace[t]
                        for qi, (start, goal) in enumerate(queries):
                            for spec in specs:
                                out = outcome(spec, rm, start, goal, true_clr, true_safe,
                                              q3.robot_radius)
                                tally.setdefault((qi, tag(spec)), []).append(out)
                                f.write(json.dumps({
                                    "row": "frame", **sbase, "stream": stream, "frame": t,
                                    # the raw frame t's flag and the gate's decision on it,
                                    # on both streams
                                    "frame_bad": bool(is_bad[t]), "gate": tr["gate"],
                                    "gate_iou": tr["gate_iou"], "n_vote": len(tr["history"]),
                                    "n_vote_bad": n_vote_bad[t],
                                    **{k: stats[k] for k in STATS_KEYS},
                                    "query": qi, "planner": spec[0], "lam": spec[1],
                                    "variant": None if spec[2] is None else spec[2].name,
                                    "tag": tag(spec), **{k: out[k] for k in FRAME_KEYS},
                                }) + "\n")
                    collided = []
                    for (qi, tg), outs in tally.items():
                        spec = next(s for s in specs if tag(s) == tg)
                        row = {"row": "sequence", **sbase, "stream": stream, "query": qi,
                               "planner": spec[0], "lam": spec[1],
                               "variant": None if spec[2] is None else spec[2].name, "tag": tg,
                               "n_builds": len(cache)}
                        for k in ("stranded", "collision", "margin_violation"):
                            row[f"{k}_frac"] = float(np.mean([o[k] for o in outs]))
                        collided.append(row["collision_frac"])
                        f.write(json.dumps(row) + "\n")
                    summary.append(f"{bad[0]}={bad[1]:g} p{p_bad:g} {stream} "
                                   f"c{np.mean(collided):.2f}")
        f.flush()
        print(f"[B] {m.name:30s} {h}x{w} {time.perf_counter() - t_map:6.1f} s  "
              + " ".join(summary))
        if q3.cooldown_s:
            time.sleep(q3.cooldown_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override q3.out_dir")
    args = ap.parse_args()

    cfg = load_q3_config(args.config)
    missing = set(cfg.q3.corruptions) - set(corruption.CORRUPTIONS)
    if missing:
        raise SystemExit(f"corruptions not in voronav.corruption: {sorted(missing)}")
    out_dir = Path(args.out or cfg.q3.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    machine = machine_info()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    warmup(cfg)
    with open(out_path, "w") as f:
        part_a(cfg, f, started, machine)
        if cfg.q3.frames.maps:
            part_b(cfg, f, started, machine)
    print(f"wrote {out_path} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
