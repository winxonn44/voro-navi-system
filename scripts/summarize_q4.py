"""Summarize a bench_q4.py results.jsonl into Markdown tables.

    python scripts/summarize_q4.py runs/bench_q4/results.jsonl > docs/q4/q4_summary.md

Mask stats are means over frames. Planning rates are over all (frame, goal)
plans of a condition and stream. Latency excludes each sequence's first
frame (it carries SAM 2's setup and prompting) and uses erosion 0 rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from plot_q4 import STREAMS, TAG_NAMES, TAGS, cond_label, cond_order, load

SHORT = {"astar": "A*", "weighted_astar@64": "wA* λ64", "voronoi[nearest]": "Vor nearest",
         "voronoi[best64]": "Vor best64"}


def pct(a: float, n: float, digits: int = 1) -> str:
    return "–" if n == 0 else f"{100 * a / n:.{digits}f}"


def mean(v, scale=1.0, fmt="{:.3f}") -> str:
    v = [x for x in v if x is not None and np.isfinite(x)]
    return "–" if not v else fmt.format(scale * float(np.mean(v)))


def counts(rows, tag, key) -> tuple[int, int]:
    h = n = 0
    for r in rows:
        v = r["out"].get(tag)
        if v:
            h += sum(bool(x) for x in v[key])
            n += len(v[key])
    return h, n


def pooled(rows, key) -> tuple[int, int]:
    h = n = 0
    for r in rows:
        for v in r["out"].values():
            h += sum(bool(x) for x in v[key])
            n += len(v[key])
    return h, n


def ms(v) -> np.ndarray:
    return 1000 * np.asarray([x for x in v if x is not None], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    rows = load(args.results)
    seqs, sams, frames = rows["sequence"], rows["sam"], rows["frame"]
    tags = [t for t in TAGS if any(t in r["out"] for r in frames)]
    conds = cond_order(frames)
    streams = [s for s in STREAMS if any(r["stream"] == s for r in frames)]
    by = defaultdict(list)
    for r in frames:
        by[(r["cond"], r["stream"])].append(r)
    raw_sam = [r for r in frames if r["source"] == "sam" and r["stream"] == "raw"]

    print("# Q4 summary\n")
    n_plans = sum(pooled([r], "collision")[1] for r in frames)
    print(f"{len(seqs)} sequences, {len(frames)} frame rows, {n_plans} graded plans.\n")

    print("## Sequences\n")
    print("| sequence | frames | camera height m | path m | yaw change ° | GT free % of grid | frames with goals | goals |")
    print("|---|---|---|---|---|---|---|---|")
    for s in seqs:
        print(f"| {s['sequence']} | {s['n_frames']} | {s['camera_height_m']:.2f} | {s['path_m']:.1f} | "
              f"{s['yaw_change_deg']:.0f} | {100 * s['gt_free_frac']:.0f} | {s['n_query_frames']} | {s['n_goals']} |")
    print()

    if raw_sam:
        print("## Mask quality per model (all sequences, erosion 0)\n")
        print("| model | rule | image IoU | image false floor % | image floor lost % | BEV IoU | BEV false free % (in view) | frames with leak ≥ 1.5 cells % |")
        print("|---|---|---|---|---|---|---|---|")
        for model in ("tiny", "small"):
            for rule in ("top_score", "largest_consistent"):
                rr = [r for r in raw_sam if r["model"] == model and r["rule"] == rule and r["erode_px"] == 0]
                if not rr:
                    continue
                deep = sum(r["n_false_free"] > 0 and r["leak_depth_max"] >= 1.5 for r in rr)
                print(f"| {model} | {rule} | {mean([r['img_free_iou'] for r in rr])} | "
                      f"{mean([r['img_false_free_frac'] for r in rr], 100, '{:.2f}')} | "
                      f"{mean([r['img_false_obstacle_frac'] for r in rr], 100, '{:.1f}')} | "
                      f"{mean([r['bev_free_iou'] for r in rr])} | "
                      f"{mean([r['bev_false_free_frac'] for r in rr], 100, '{:.2f}')} | {pct(deep, len(rr), 0)} |")
        print()

        print("## Mask quality per sequence and condition\n")
        print("| sequence | condition | image IoU | min image IoU | image false floor % | image floor lost % | BEV IoU | BEV false free % | leak depth p90 (cells) | boundary cells leaked % | small obstacles erased |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted({r["sequence"] for r in raw_sam}):
            for c in [c for c in conds if c.startswith("sam")]:
                rr = [r for r in raw_sam if r["sequence"] == s and r["cond"] == c]
                if not rr:
                    continue
                print(f"| {s} | {cond_label(c)} | {mean([r['img_free_iou'] for r in rr])} | "
                      f"{min(r['img_free_iou'] for r in rr):.3f} | "
                      f"{mean([r['img_false_free_frac'] for r in rr], 100, '{:.2f}')} | "
                      f"{mean([r['img_false_obstacle_frac'] for r in rr], 100, '{:.1f}')} | "
                      f"{mean([r['bev_free_iou'] for r in rr])} | "
                      f"{mean([r['bev_false_free_frac'] for r in rr], 100, '{:.2f}')} | "
                      f"{mean([r['leak_depth_p90'] for r in rr], 1, '{:.1f}')} | "
                      f"{mean([r['boundary_leak_frac'] for r in rr], 100, '{:.1f}')} | "
                      f"{sum(r['n_erased'] for r in rr)} |")
        print()

        print("## SAM's multimask choice on the prompt frame\n")
        print("| sequence | model | rule | chosen | candidate scores | candidate areas (px) |")
        print("|---|---|---|---|---|---|")
        for r in sams:
            for ch in r.get("choices") or []:
                print(f"| {r['sequence']} | {r['model']} | {r['rule']} | {ch['index']} | "
                      f"{', '.join(f'{x:.2f}' for x in ch['scores'])} | "
                      f"{', '.join(str(a) for a in ch['areas'])} |")
        print()

    for key, title in (("collision", "Collision % (path enters a true obstacle or unseen cell)"),
                       ("margin_violation", "Margin violation % (true clearance < robot radius)")):
        print(f"## {title}\n")
        print("Raw → each filtered stream (" + ", ".join(STREAMS[s] for s in streams[1:]) + ").\n")
        print("| condition | " + " | ".join(SHORT[t] for t in tags) + " |")
        print("|---|" + "---|" * len(tags))
        for c in conds:
            cells = []
            for t in tags:
                vals = [pct(*counts(by[(c, s)], t, key)) for s in streams]
                cells.append(" → ".join(vals))
            print(f"| {cond_label(c)} | " + " | ".join(cells) + " |")
        print()

    print("## Stranded % (all planners), raw → filtered\n")
    print("| condition | " + " | ".join(STREAMS[s] for s in streams) + " | plans per stream |")
    print("|---|" + "---|" * len(streams) + "---|")
    for c in conds:
        cells = [pct(*pooled(by[(c, s)], "stranded")) for s in streams]
        print(f"| {cond_label(c)} | " + " | ".join(cells) + f" | {pooled(by[(c, 'raw')], 'stranded')[1]} |")
    print()

    print("## Why plans strand (all planners)\n")
    print("*start unsafe*: every goal of the frame stranded at an endpoint for that planner, i.e. the start "
          "cell (the safe cell nearest the robot, just past the camera's blind zone) is not safe on the "
          "estimated grid. *goal unsafe*: another endpoint strand. *no path*: both endpoints safe, no route.\n")
    print("| condition | stream | start unsafe % of plans | goal unsafe % | no path % |")
    print("|---|---|---|---|---|")
    for c in conds:
        for s in streams:
            st = go = nop = n = 0
            for r in by[(c, s)]:
                for v in r["out"].values():
                    reasons = v["strand_reason"]
                    n += len(reasons)
                    if reasons and all(x == "endpoint" for x in reasons):
                        st += len(reasons)
                    else:
                        go += sum(x == "endpoint" for x in reasons)
                    nop += sum(x == "no_path" for x in reasons)
            if st + go + nop:
                print(f"| {cond_label(c)} | {STREAMS[s]} | {pct(st, n)} | {pct(go, n)} | {pct(nop, n)} |")
    print()

    inj = [r for r in frames if r["source"] == "inject" and r["stream"] != "raw"]
    if inj:
        print("## Gate detection on injected bad frames (GT stream + Q3 corruption on random frames)\n")
        print("| injected | stream | bad frames | rejected | precision | recall |")
        print("|---|---|---|---|---|---|")
        for c in sorted({r["cond"] for r in inj}):
            for s in streams[1:]:
                rr = [r for r in inj if r["cond"] == c and r["stream"] == s and r["gate"] != "first"]
                bad = sum(r["frame_bad"] for r in rr)
                rej = sum(r["gate"] == "reject" for r in rr)
                tp = sum(r["gate"] == "reject" and r["frame_bad"] for r in rr)
                print(f"| {cond_label(c)} | {STREAMS[s]} | {bad} | {rej} | {pct(tp, rej, 0)} | {pct(tp, bad, 0)} |")
        print()

    if raw_sam:
        print("## Gates on SAM frames\n")
        print("A raw frame is *unsafe* if any planner collided on it. Rejected = the gate kept it out of the vote.\n")
        print("| condition | stream | frames | rejected % | unsafe raw frames | of those rejected |")
        print("|---|---|---|---|---|---|")
        raw_unsafe = {(r["sequence"], r["cond"], r["frame"]): pooled([r], "collision")[0] > 0
                      for r in raw_sam}
        for c in [c for c in conds if c.startswith("sam")]:
            for s in streams[1:]:
                rr = [r for r in by[(c, s)]]
                rej = [r["gate"] == "reject" for r in rr]
                uns = [raw_unsafe.get((r["sequence"], c, r["frame"]), False) for r in rr]
                caught = sum(a and b for a, b in zip(rej, uns))
                print(f"| {cond_label(c)} | {STREAMS[s]} | {len(rr)} | {pct(sum(rej), len(rr), 0)} | "
                      f"{sum(uns)} | {caught} |")
        print()

        print("## End-to-end latency per frame (ms, M4 MacBook Air, erosion 0, frame 0 excluded)\n")
        lat = [r for r in raw_sam if r["erode_px"] == 0 and r["frame"] > 0 and r["lat"]["sam_s"] is not None]
        print("| model | SAM 2 median | SAM 2 p90 | bird's-eye | roadmap build | " +
              " | ".join(f"plan {SHORT[t]}" for t in tags) + " | end-to-end median | end-to-end p90 | fps |")
        print("|---|---|---|---|---|" + "---|" * len(tags) + "---|---|---|")
        for model in ("tiny", "small"):
            rr = [r for r in lat if r["model"] == model]
            if not rr:
                continue
            sam = ms([r["lat"]["sam_s"] for r in rr])
            plans = [f"{np.median(ms([r['lat']['plan_s'].get(t) for r in rr])):.1f}" for t in tags]
            plan_t = "voronoi[best64]" if "voronoi[best64]" in tags else tags[0]
            tot = ms([r["lat"]["sam_s"] + r["lat"]["bev_s"] + r["lat"]["build_s"]
                      + (r["lat"]["plan_s"].get(plan_t) or 0) for r in rr])
            print(f"| {model} | {np.median(sam):.0f} | {np.percentile(sam, 90):.0f} | "
                  f"{np.median(ms([r['lat']['bev_s'] for r in rr])):.1f} | "
                  f"{np.median(ms([r['lat']['build_s'] for r in rr])):.1f} | " + " | ".join(plans) +
                  f" | {np.median(tot):.0f} | {np.percentile(tot, 90):.0f} | {1000 / np.median(tot):.2f} |")
        print(f"\nEnd-to-end = SAM 2 + bird's-eye + roadmap build + one Voronoi best64 plan. "
              f"The temporal filter adds {np.median(ms([r['lat']['filter_s'] for r in frames if r['stream'] != 'raw' and r['source'] == 'sam'])):.1f} ms median.")


if __name__ == "__main__":
    main()
