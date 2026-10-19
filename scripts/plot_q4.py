"""Plot the Q4 study from a bench_q4.py results.jsonl.

    python scripts/plot_q4.py runs/bench_q4/results.jsonl --out docs/q4

  q4_mask_quality.png  per sequence and SAM condition: image IoU and false
                       floor rate against GT, and the same on the BEV grid
  q4_on_q3.png         SAM's frames placed on Q3's collision curves: Q3's
                       collision rate vs boundary_shift (px) per planner,
                       and Q4's collision rate per frame binned by how deep
                       its false floor reaches into true obstacles (px)
  q4_signed_error.png  distribution of SAM's BEV errors by signed distance
                       to the true floor boundary (< 0 = floor leaking into
                       obstacles, > 0 = floor lost), per model and erosion
  q4_outcomes.png      collision / margin-violation / stranded rates per
                       planner and condition, raw vs filtered streams
  q4_gates.png         the temporal filter's gates on injected bad frames:
                       detection precision / recall, and the scores they saw
  q4_latency.png       end-to-end per-frame latency on the laptop, by stage
Rates are over all (frame, goal) plans; bars = ±1 binomial standard error.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_q2 import INK, MUTED, PLANNERS, SURFACE, style

ROOT = Path(__file__).resolve().parent.parent
TAGS = ["astar", "weighted_astar@64", "voronoi[nearest]", "voronoi[best64]"]
TAG_NAMES = {"astar": "Grid A*", "weighted_astar@64": "Weighted A*, λ=64",
             "voronoi[nearest]": "Voronoi, nearest", "voronoi[best64]": "Voronoi, best64"}
TAG_COLORS = {"astar": PLANNERS["astar"][0], "weighted_astar@64": PLANNERS["weighted_astar"][0],
              "voronoi[nearest]": "#8fb7e8", "voronoi[best64]": PLANNERS["voronoi"][0]}
STREAMS = {"raw": "raw frames", "vote_iou": "vote, IoU gate", "vote_leak": "vote, leak gate",
           "vote_both": "vote, both gates"}
# SAM conditions: model -> hue, rule -> lightness, erosion -> hatch
MODEL_COLORS = {"tiny": ("#2a78d6", "#8fb7e8"), "small": ("#d6455d", "#eda0ad")}
RULE_SHORT = {"top_score": "top", "largest_consistent": "largest"}
GOOD, BAD = "#8a897f", "#d6455d"


def load(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["row"]].append(r)
    return out


def cond_order(frames: list[dict]) -> list[str]:
    conds = {r["cond"]: r for r in frames}

    def key(c):
        r = conds[c]
        src = {"gt": 0, "inject": 1, "sam": 2}[r["source"]]
        return (src, ["tiny", "small"].index(r["model"]) if r.get("model") else 0,
                r.get("rule") or "", r.get("erode_px") or 0, c)
    return sorted(conds, key=key)


def cond_label(c: str) -> str:
    if c == "gt":
        return "GT (control)"
    if c.startswith("inject:"):
        return "GT + " + c.split(":", 1)[1].replace("boundary_shift", "shift").replace(
            "thin_obstacle_miss", "thin miss")
    _, model, rule, e = c.replace("sam2.1-", "sam/").split("/")
    return f"{model} {RULE_SHORT.get(rule, rule)} {e}"


def sam_style(r: dict) -> dict:
    dark, light = MODEL_COLORS[r["model"]]
    return dict(color=dark if r["rule"] == "largest_consistent" else light,
                hatch="//" if r["erode_px"] else None, edgecolor=SURFACE)


def rate(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, 0.0
    p = hits / n
    return 100 * p, 100 * np.sqrt(p * (1 - p) / n)


def tally(frames: list[dict], key: str) -> dict[str, list[int]]:
    """tag -> [hits, n] of an outcome key over all goals of `frames`."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in frames:
        for t, v in r["out"].items():
            out[t][0] += sum(bool(x) for x in v[key])
            out[t][1] += len(v[key])
    return out


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / name)


# ---------------------------------------------------------------------------


def mask_quality(frames, out: Path) -> None:
    sam = [r for r in frames if r["source"] == "sam" and r["stream"] == "raw"]
    if not sam:
        return
    seqs = sorted({r["sequence"] for r in sam})
    conds = [c for c in cond_order(sam)]
    first = {r["cond"]: r for r in sam}
    panels = [("img_free_iou", "Image: floor IoU vs GT", "IoU", 1),
              ("img_false_free_frac", "Image: false floor", "% of true non-floor pixels", 100),
              ("bev_free_iou", "Bird's-eye grid: free-space IoU", "IoU", 1),
              ("bev_false_free_frac", "Bird's-eye grid: false free", "% of true obstacle cells in view", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    width = 0.8 / len(conds)
    for ax, (key, title, ylabel, scale) in zip(axes.flat, panels):
        for j, c in enumerate(conds):
            ys, es = [], []
            for s in seqs:
                v = [r[key] for r in sam if r["sequence"] == s and r["cond"] == c and r[key] is not None]
                ys.append(scale * np.mean(v) if v else np.nan)
                es.append(scale * np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0)
            x = np.arange(len(seqs)) + (j - (len(conds) - 1) / 2) * width
            ax.bar(x, ys, width, yerr=es, label=cond_label(c), error_kw=dict(ecolor=MUTED, lw=0.8),
                   **sam_style(first[c]))
        ax.set_xticks(np.arange(len(seqs)), [s.replace("_", "\n") for s in seqs])
        if "iou" in key:
            lo = min(np.nanmin([r[key] for r in sam if r[key] is not None]), 0.8)
            ax.set_ylim(max(0, lo - 0.05), 1.0)
        style(ax, title, "", ylabel)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False, title="model, rule, erosion (px)",
                      title_fontsize=8)
    fig.suptitle("SAM 2 floor masks against TartanGround ground truth (mean over frames ± s.e.)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, out, "q4_mask_quality.png")


def q3_curves(q3_path: Path) -> dict:
    """(group, tag) -> {level: [collisions, n]} for Q3 Part A boundary_shift >= 0."""
    curves: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if not q3_path.exists():
        return curves
    opener = gzip.open if q3_path.suffix == ".gz" else open
    with opener(q3_path, "rt") as f:
        for line in f:
            if '"row": "query"' not in line or '"boundary_shift"' not in line:
                continue
            r = json.loads(line)
            if r["level"] < 0:
                continue
            c = curves[(r["group"], r["tag"])][r["level"]]
            c[0] += bool(r["collision"])
            c[1] += 1
    return curves


def on_q3(frames, q3_path: Path, out: Path) -> None:
    sam = [r for r in frames if r["source"] == "sam" and r["stream"] == "raw"]
    if not sam:
        return
    curves = q3_curves(q3_path)
    tags = [t for t in TAGS if any(t in r["out"] for r in sam)]
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 4.2), facecolor=SURFACE,
                             squeeze=False, sharey=True)
    # Q4 x: deepest false-free cell of the frame, in Q3 shift units: a shift
    # of +L frees cells up to depth L - 0.5, so depth d ~ shift d + 0.5
    for ax, t in zip(axes[0], tags):
        for group, ls in (("structural", "-"), ("cluttered", ":")):
            c = curves.get((group, t))
            if c:
                lv = sorted(c)
                ax.plot(lv, [100 * c[x][0] / c[x][1] for x in lv], ls, color=MUTED, lw=1.5,
                        marker="o", ms=3, label=f"Q3 boundary shift, {group}")
        for model, (dark, _) in MODEL_COLORS.items():
            rows = [r for r in sam if r["model"] == model]
            if not rows:
                continue
            bins: dict[int, list[int]] = defaultdict(lambda: [0, 0])
            for r in rows:
                b = int(np.clip(np.ceil(r["leak_depth_max"] + 0.5 - 1e-9), 0, 6)) if r["n_false_free"] else 0
                v = r["out"].get(t)
                if v:
                    bins[b][0] += sum(v["collision"])
                    bins[b][1] += len(v["collision"])
            xs = sorted(bins)
            ys = [rate(*bins[x]) for x in xs]
            ax.errorbar(np.array(xs) + (0.08 if model == "small" else -0.08), [y[0] for y in ys],
                        yerr=[y[1] for y in ys], fmt="s", color=dark, ms=6, capsize=2,
                        label=f"Q4 SAM {model} frames (all rules, erosions)")
            for x in xs:
                ax.annotate(f"{bins[x][1]}", (x, 0), textcoords="offset points", xytext=(0, 8 if model == "tiny" else 17),
                            ha="center", fontsize=6, color=dark)
        style(ax, TAG_NAMES[t], "leak depth (px): Q3 shift level, Q4 deepest false-free cell + 0.5",
              "collision % of plans" if t == tags[0] else "")
        ax.set_xlim(-0.5, 6.5)
        ax.xaxis.label.set_fontsize(8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("SAM 2's errors on Q3's collision curves (Q4 bins: 6 = 6 px or deeper; small numbers = plans per bin)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, out, "q4_on_q3.png")


def signed_error(frames, out: Path) -> None:
    sam = [r for r in frames if r["source"] == "sam" and r["stream"] == "raw"]
    if not sam:
        return
    centers = np.arange(-10, 11)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE, sharey=True)
    for ax, e in zip(axes, sorted({r["erode_px"] for r in sam})[:2]):
        for model, (dark, light) in MODEL_COLORS.items():
            for rule in RULE_SHORT:
                rows = [r for r in sam if r["model"] == model and r["rule"] == rule and r["erode_px"] == e]
                if not rows:
                    continue
                h = np.sum([r["signed_hist"] for r in rows], axis=0) / len(rows)
                ax.step(centers, np.maximum(h, 1e-2), where="mid",
                        color=dark if rule == "largest_consistent" else light,
                        ls="-" if rule == "largest_consistent" else "--",
                        label=f"{model}, {RULE_SHORT[rule]}")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.text(-9.8, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1, "← floor leaks into obstacles",
                fontsize=8, color=BAD)
        ax.text(9.8, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1, "floor lost →", fontsize=8,
                color=MUTED, ha="right")
        style(ax, f"BEV errors by signed boundary distance, erosion {e} px",
              "signed distance to the true floor boundary (cells; clipped at ±10)",
              "wrong cells per frame" if e == 0 else "")
    axes[0].legend(fontsize=8, frameon=False, loc="upper center")
    save(fig, out, "q4_signed_error.png")


def outcomes(frames, out: Path) -> None:
    conds = cond_order(frames)
    streams = [s for s in STREAMS if any(r["stream"] == s for r in frames)]
    keys = [("collision", "Collision"), ("margin_violation", "Margin violation"),
            ("stranded", "Stranded")]
    by = defaultdict(list)
    for r in frames:
        by[(r["cond"], r["stream"])].append(r)
    tags = [t for t in TAGS if any(t in r["out"] for r in frames)]
    fig, axes = plt.subplots(len(keys), len(streams), figsize=(5.6 * len(streams), 3.4 * len(keys)),
                             facecolor=SURFACE, squeeze=False, sharey="row")
    width = 0.8 / len(tags)
    for i, (key, title) in enumerate(keys):
        for j, s in enumerate(streams):
            ax = axes[i, j]
            for k, t in enumerate(tags):
                ys, es = [], []
                for c in conds:
                    hits, n = tally(by[(c, s)], key)[t]
                    y, e = rate(hits, n)
                    ys.append(y)
                    es.append(e)
                x = np.arange(len(conds)) + (k - (len(tags) - 1) / 2) * width
                ax.bar(x, ys, width, yerr=es, color=TAG_COLORS[t], label=TAG_NAMES[t],
                       error_kw=dict(ecolor=MUTED, lw=0.6))
            ax.set_xticks(np.arange(len(conds)), [cond_label(c) for c in conds], rotation=60,
                          ha="right", fontsize=7)
            style(ax, f"{title}: {STREAMS[s]}", "", "% of plans" if j == 0 else "")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Planning on each condition's grid, graded on the ground-truth grid of the same frame",
                 x=0.01, ha="left", color=INK, fontsize=12)
    save(fig, out, "q4_outcomes.png")


def gates(frames, out: Path) -> None:
    inj = [r for r in frames if r["source"] == "inject" and r["stream"].startswith("vote_")
           and r["gate"] in ("accept", "reject", "reset")]
    if not inj:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    # detection, per injected model and gate (one row per frame -> dedupe goals: rows are per frame)
    labels, prec, rec = [], [], []
    for c in sorted({r["cond"] for r in inj}):
        for s in sorted({r["stream"] for r in inj}):
            rows = [r for r in inj if r["cond"] == c and r["stream"] == s]
            bad = np.array([r["frame_bad"] for r in rows])
            rej = np.array([r["gate"] == "reject" for r in rows])
            labels.append(f"{cond_label(c)}\n{STREAMS[s]}")
            prec.append(100 * (rej & bad).sum() / rej.sum() if rej.any() else np.nan)
            rec.append(100 * (rej & bad).sum() / bad.sum() if bad.any() else np.nan)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, prec, 0.4, color=MUTED, label="precision")
    axes[0].bar(x + 0.2, rec, 0.4, color=BAD, label="recall")
    axes[0].set_xticks(x, labels, fontsize=7)
    axes[0].legend(fontsize=8, frameon=False)
    style(axes[0], "Gate detection of injected bad frames", "", "%")
    for ax, key, s, title in ((axes[1], "gate_iou", "vote_iou", "IoU with last accepted grid"),
                              (axes[2], "gate_leak", "vote_leak", "Leak score (new free near boundary)")):
        rows = [r for r in inj if r["stream"] == s and r[key] is not None]
        if not rows:
            continue
        for k, c in enumerate(sorted({r["cond"] for r in rows})):
            rr = [r for r in rows if r["cond"] == c]
            for b, col in ((False, GOOD), (True, BAD)):
                v = [r[key] for r in rr if r["frame_bad"] == b]
                ax.scatter(np.full(len(v), k) + (0.15 if b else -0.15)
                           + np.random.default_rng(k).uniform(-0.08, 0.08, len(v)), v, s=6,
                           color=col, alpha=0.6, label=("bad frame" if b else "good frame") if k == 0 else None)
        ax.set_xticks(range(len({r["cond"] for r in rows})),
                      [cond_label(c) for c in sorted({r["cond"] for r in rows})], fontsize=8)
        style(ax, title, "", key)
        ax.legend(fontsize=8, frameon=False)
    save(fig, out, "q4_gates.png")


def latency(frames, out: Path) -> None:
    sam = [r for r in frames if r["source"] == "sam" and r["stream"] == "raw"
           and r["lat"]["sam_s"] is not None and r["erode_px"] == 0]
    if not sam:
        return
    tag = "voronoi[best64]" if any("voronoi[best64]" in r["lat"]["plan_s"] for r in sam) else None
    stages = [("sam_s", "SAM 2", "#2a78d6"), ("bev_s", "bird's-eye", "#1baf7a"),
              ("build_s", "roadmap (JFA + GVD)", "#eb6834"), ("plan_s", "plan (Voronoi best64)", "#d6455d")]
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    models = [m for m in ("tiny", "small") if any(r["model"] == m for r in sam)]
    for i, m in enumerate(models):
        rows = [r for r in sam if r["model"] == m and r["frame"] > 0]  # frame 0 includes setup
        parts = {}
        for key, _, _ in stages:
            if key == "plan_s":
                v = [r["lat"]["plan_s"].get(tag) for r in rows]
                v = [x for x in v if x is not None]
            else:
                v = [r["lat"][key] for r in rows]
            parts[key] = float(np.median(v)) * 1000 if v else 0.0
        left = 0.0
        for key, name, col in stages:
            ax.barh(i, parts[key], left=left, color=col, label=name if i == 0 else None, height=0.55)
            left += parts[key]
        tot = [1000 * (r["lat"]["sam_s"] + r["lat"]["bev_s"] + r["lat"]["build_s"]
                       + (r["lat"]["plan_s"].get(tag) or 0)) for r in rows]
        ax.text(left + 8, i, f"median {np.median(tot):.0f} ms, p90 {np.percentile(tot, 90):.0f} ms"
                f"  ({1000 / np.median(tot):.1f} fps)", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(models)), [f"SAM 2.1 {m}" for m in models])
    ax.set_xlim(0, ax.get_xlim()[1] * 1.35)
    style(ax, "End-to-end latency per frame on a fanless M4 MacBook Air (stage medians)",
          "ms per frame", "")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    save(fig, out, "q4_latency.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "q4")
    ap.add_argument("--q3", type=Path, default=ROOT / "docs" / "runs" / "q3_results.jsonl.gz",
                    help="Q3 results for the curves in q4_on_q3.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.results)
    frames = rows["frame"]
    mask_quality(frames, args.out)
    on_q3(frames, args.q3, args.out)
    signed_error(frames, args.out)
    outcomes(frames, args.out)
    gates(frames, args.out)
    latency(frames, args.out)


if __name__ == "__main__":
    main()
