"""Q4 benchmark harness: is real SAM 2 on real first-person video accurate
enough to plan safely on, and how fast does it run offline?

    python scripts/bench_q4.py --config configs/debug_q4.yaml   # 2 sequences x 8 frames
    python scripts/bench_q4.py --config configs/bench_q4.yaml   # the study, run plugged in

Per sequence (TartanGround, voronav.vision.dataset):
  1. Ground truth: every frame's GT floor mask -> mask_to_bev with that
     frame's camera -> GT BEV (True = obstacle or unseen). The grader uses
     its exact clearance and safe mask (free, clearance >= robot radius).
  2. Queries: start = the GT-safe cell nearest the robot's own cell (the
     robot's cell itself is in the camera's blind zone, below the image, so
     it is unseen = obstacle in every BEV); goals = `goals_per_frame` seeded
     cells of the start's reachable GT-safe region (4-connected, Q3's move
     rule), `goal_dist_m` away from the start.
  3. SAM conditions: model x selection rule runs SAM 2 once per sequence
     (CachedSegmenter); each erosion level is a post-process on those masks.
     Per frame: image-space mask stats vs GT, the mask through mask_to_bev,
     BEV stats vs the GT BEV, and the BEV error in Q3's terms (signed
     boundary error: how deep false floor reaches into true obstacles).
  4. Streams: "raw" (plan on each frame's own BEV) and one "vote_<gate>" per
     configured gate: Q3's majority vote over the last accepted BEVs warped
     into the current frame with the ground-truth relative pose
     (voronav.vision.egomotion; a real robot would use odometry).
  5. Planning: A*, weighted A*, Voronoi variants on each stream's BEV, graded
     on the GT BEV with Q3's outcome definitions (bench_q3.outcome).
  6. Controls: the GT BEV itself as a condition (must never collide), and
     optionally GT streams with Q3 corruptions injected on random frames
     (q4.inject) so the gates are scored on known leaks under real motion.

Rows ("row" field):
  sequence  one per sequence: camera, floor fit, frame count, machine
  sam       one per (sequence, model, rule): SAM 2 per-frame wall time,
            the multimask choice on the prompt frames, cache hit or miss
  frame     one per (sequence, condition, stream, frame): mask stats
            (image and BEV), Q3-terms leak stats, the gate's decision,
            roadmap build stats, latency parts, and per planner tag the
            outcome arrays over the frame's goals (collision,
            margin_violation, stranded, strand_reason, min_clearance_px,
            length_px, plan_s)
Outcomes, graded on the GT BEV (exact SciPy EDT):
  stranded          an endpoint is unsafe on the estimated BEV, or no path
  collision         the path enters a true obstacle-or-unseen cell
  margin_violation  true clearance < robot radius somewhere on the path
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from bench_jfa import machine_info
from bench_q3 import outcome, plan, stable_seed, tag
from voronav import corruption
from voronav.config import Q4Config, load_q4_config, to_dict
from voronav.planning import build_roadmap, exact_clearance
from voronav.planning.grid_search import REACHABILITY
from voronav.vision.bev import bev_shape, fov_mask, mask_to_bev, robot_cell
from voronav.vision.dataset import list_sequences, load_sequence
from voronav.vision.egomotion import AlignedTemporalFilter, ground_poses
from voronav.vision.segment import CachedSegmenter, Prompts, default_floor_prompts, erode

ROOT = Path(__file__).resolve().parent.parent
OUT_KEYS = ("collision", "margin_violation", "stranded", "strand_reason",
            "min_clearance_px", "length_px", "plan_s")
IMG_KEYS = ("free_iou", "false_free_frac", "false_obstacle_frac", "boundary_error_px")


# ---------------------------------------------------------------------------
# Planners and roadmaps
# ---------------------------------------------------------------------------


def planner_specs(cfg: Q4Config) -> list[tuple[str, float | None, object]]:
    """(planner, lambda, Voronoi variant config), as bench_q3.planner_specs."""
    q4 = cfg.q4
    specs = []
    for planner in q4.planners:
        if planner == "voronoi":
            specs += [("voronoi", None, v) for v in q4.voronoi_variants]
        elif planner == "astar":
            specs.append(("astar", None, None))
        elif planner == "weighted_astar":
            specs += [("weighted_astar", lam, None) for lam in q4.lambdas]
    return specs


def build(cfg: Q4Config, occ: np.ndarray):
    r = cfg.roadmap
    return build_roadmap(occ, robot_radius=cfg.q4.robot_radius, min_angle_deg=r.min_angle_deg,
                         raster_slack_px=r.raster_slack_px, min_piece_px=r.min_piece_px,
                         bridge_lam=r.bridge_lam, backend=r.backend, variant=r.variant)


def warmup(cfg: Q4Config) -> None:
    """Compile the Numba kernels before anything is timed."""
    h, w = bev_shape(cfg.q4.px_per_m, tuple(cfg.q4.extent_m))
    occ = np.ones((h, w), bool)
    occ[10:-10, 10:-10] = False
    occ[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5] = True
    rm = build(cfg, occ)
    for spec in planner_specs(cfg):
        plan(spec, rm, (15, 15), (h - 15, w - 15))


# ---------------------------------------------------------------------------
# Ground truth and queries
# ---------------------------------------------------------------------------


def truth(occ: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    clr = exact_clearance(occ)
    return clr, ~occ & (clr >= radius)


def pick_queries(true_safe: np.ndarray, robot: tuple[int, int], n: int, dmin_px: float,
                 dmax_px: float, rng: np.random.Generator):
    """start = the GT-safe cell nearest the robot cell; goals = up to n
    distinct cells of the start's reachable safe region, dmin..dmax px from
    the start. (None, []) when no safe cell exists."""
    cells = np.argwhere(true_safe)
    if len(cells) == 0:
        return None, []
    d = np.hypot(cells[:, 0] - robot[0], cells[:, 1] - robot[1])
    s = cells[int(np.argmin(d))]
    regions, _ = label(true_safe, structure=REACHABILITY)
    pix = np.argwhere(regions == regions[s[0], s[1]])
    dist = np.hypot(*(pix - s).T)
    cand = pix[(dist >= dmin_px) & (dist <= dmax_px)]
    k = min(n, len(cand))
    goals = cand[rng.choice(len(cand), size=k, replace=False)] if k else []
    return (int(s[0]), int(s[1])), [(int(g[0]), int(g[1])) for g in goals]


# ---------------------------------------------------------------------------
# Error statistics in Q3's terms
# ---------------------------------------------------------------------------

HIST_EDGES = np.arange(-10.5, 11.0, 1.0)  # signed error bins, px; < 0 = leak into obstacles


def q3_terms(true_occ: np.ndarray, est_occ: np.ndarray, sdf: np.ndarray, seen: np.ndarray) -> dict:
    """The estimate's error as Q3's error models describe it.

    sdf: corruption.signed_floor_distance(true_occ) (positive on floor).
    leak_depth_*: for cells the estimate calls floor but the truth calls
      obstacle, how deep into the obstacle they sit (px; a boundary_shift of
      +L frees cells up to depth ~L - 0.5).
    shrink_depth_*: the same for floor marked obstacle, inside the field of
      view only (outside it, both grids are unseen).
    boundary_leak_frac: share of true obstacle-boundary cells inside the
      view that the estimate frees (boundary_shift +1 would free all).
    n_erased, erased_max_width_px: true obstacle components fully inside the
      view (not touching unseen space) that vanish from the estimate,
      thin_obstacle_miss-style, and the widest one (Q3's width).
    signed_hist: counts of mismatched cells by signed true-floor distance,
      bins of 1 px from -10 to +10 (clipped).
    """
    ff = ~est_occ & true_occ
    fo = est_occ & ~true_occ & seen
    leak = -sdf[ff]
    shrink = sdf[fo]
    bnd = true_occ & binary_dilation(~true_occ, structure=REACHABILITY) & seen
    out = {
        "n_false_free": int(ff.sum()),
        "n_false_obstacle": int(fo.sum()),
        "leak_depth_p50": float(np.percentile(leak, 50)) if leak.size else 0.0,
        "leak_depth_p90": float(np.percentile(leak, 90)) if leak.size else 0.0,
        "leak_depth_max": float(leak.max()) if leak.size else 0.0,
        "shrink_depth_p50": float(np.percentile(shrink, 50)) if shrink.size else 0.0,
        "shrink_depth_p90": float(np.percentile(shrink, 90)) if shrink.size else 0.0,
        "boundary_leak_frac": float((bnd & ~est_occ).sum() / bnd.sum()) if bnd.any() else 0.0,
    }
    # small obstacles: true obstacle components that never touch unseen
    # space or the grid edge (an unseen cell is obstacle too, so anything
    # touching it is part of the big "unknown" mass)
    comp, n = label(true_occ, structure=np.ones((3, 3), bool))
    erased, widths = 0, []
    if n:
        touches = np.zeros(n + 1, bool)
        touches[np.unique(comp[true_occ & ~seen])] = True
        edge = np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]])
        touches[edge] = True
        touches[0] = True
        alive = np.zeros(n + 1, bool)
        alive[np.unique(comp[true_occ & est_occ])] = True
        gone = ~touches & ~alive
        gone[0] = False
        if gone.any():
            depth = distance_transform_edt(true_occ)
            max_depth = np.zeros(n + 1)
            np.maximum.at(max_depth, comp.ravel(), depth.ravel())
            widths = (2 * max_depth[gone]).tolist()
            erased = int(gone.sum())
    out["n_erased"] = erased
    out["erased_max_width_px"] = float(max(widths)) if widths else 0.0
    vals = np.concatenate([-leak, shrink]) if (leak.size or shrink.size) else np.zeros(0)
    hist, _ = np.histogram(np.clip(vals, -10, 10), bins=HIST_EDGES)
    out["signed_hist"] = hist.tolist()
    return out


def bev_stats(true_occ: np.ndarray, est_occ: np.ndarray, seen: np.ndarray, sdf: np.ndarray) -> dict:
    """corruption.mask_error_stats on the BEV, plus the Q3-terms stats.
    false_free_frac here is over true obstacle cells INSIDE the view
    (outside it both grids are unseen), so it is comparable across cameras."""
    s = corruption.mask_error_stats(true_occ, est_occ)
    n_obs_seen = int((true_occ & seen).sum())
    out = {"bev_" + k: s[k] for k in ("free_iou", "false_obstacle_frac", "boundary_error_px")}
    out["bev_false_free_frac"] = s["n_false_free"] / n_obs_seen if n_obs_seen else 0.0
    out["bev_free_frac"] = float((~est_occ).mean())
    out.update(q3_terms(true_occ, est_occ, sdf, seen))
    return out


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------


class LazySam2:
    """Sam2Floor(model, rule, erode_px=0) built only on a cache miss, with
    Sam2Floor's own cache_tag so CachedSegmenter keys match."""

    def __init__(self, model: str, rule: str, precision: str | None):
        self.model, self.rule, self.precision = model, rule, precision
        self.inner = None
        self.last_frame_s: list[float] | None = None
        self.last_choices: list[dict] | None = None

    def _precision(self) -> str:
        if self.precision:
            return self.precision
        from voronav.vision.segment import default_device
        return "bf16" if default_device() == "mps" else "fp32"

    @property
    def cache_tag(self) -> str:
        return f"sam2.1-{self.model}/{self.rule}/erode0/{self._precision()}"

    def segment_video(self, frames, prompts):
        from voronav.vision.segment import Sam2Floor
        self.inner = Sam2Floor(model=self.model, select=self.rule, erode_px=0,
                               precision=self.precision)
        masks = self.inner.segment_video(frames, prompts)
        self.last_frame_s, self.last_choices = self.inner.last_frame_s, self.inner.last_choices
        self.inner = None  # free the network (and MPS memory) between runs
        return masks

    def segment_image(self, frame, prompts):
        raise NotImplementedError


class FakeFloor:
    """Test stand-in, no weights: the GT floor masks with a deterministic
    error, the floor grown by 2 px on every 4th frame (a leak) and shrunk by
    1 px elsewhere. Timing is fake (1 ms per frame)."""

    def __init__(self, model: str, rule: str, gt_masks):
        self.model, self.rule, self.gt = model, rule, gt_masks
        self.last_frame_s = None
        self.last_choices = None

    @property
    def cache_tag(self) -> str:
        return f"fake-{self.model}/{self.rule}/erode0"

    def segment_video(self, frames, prompts):
        out = []
        for i in range(len(frames)):
            m = self.gt[i]
            out.append(binary_dilation(m, iterations=2) if i % 4 == 3 else erode(m, 1))
        self.last_frame_s = [0.001] * len(frames)
        self.last_choices = [{"index": 0, "scores": [1.0], "areas": [int(out[0].sum())]}]
        return out

    def segment_image(self, frame, prompts):
        raise NotImplementedError


def run_sam(cfg: Q4Config, seq, frames, gt_masks, model: str, rule: str, prompts: Prompts,
            fake: bool, cache_dir: Path) -> tuple[list[np.ndarray], dict]:
    """Masks (erosion 0) and a timing / choice record for one (model, rule)."""
    inner = FakeFloor(model, rule, gt_masks) if fake else LazySam2(model, rule, cfg.q4.precision)
    seg = CachedSegmenter(inner, cache_dir)
    sidecar = seg.path(frames, prompts, seq.name).with_suffix(".info.json")
    t0 = time.perf_counter()
    masks = seg.segment_video(frames, prompts, sequence=seq.name)
    wall = time.perf_counter() - t0
    if seg.misses:
        info = {"frame_s": inner.last_frame_s, "choices": inner.last_choices, "wall_s": wall,
                "cache_tag": seg.cache_tag}
        sidecar.write_text(json.dumps(info))
    elif sidecar.exists():
        info = json.loads(sidecar.read_text())
    else:  # cached by something else (e.g. a smoke test) without timing
        info = {"frame_s": None, "choices": None, "wall_s": None, "cache_tag": seg.cache_tag}
    info["cache_hit"] = bool(seg.hits)
    return masks, info


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def streams(cfg: Q4Config, bevs: list[np.ndarray], seen: list[np.ndarray], gposes: np.ndarray):
    """{"raw": (grids, trace, times), "vote_<gate>": ...}; trace entries are
    the gate's decision per frame (None for raw), times the filter's cost."""
    fc = cfg.q4.filter
    out = {"raw": (bevs, [None] * len(bevs), [0.0] * len(bevs))}
    for gate in fc.gates:
        filt = AlignedTemporalFilter(cfg.q4.px_per_m, tuple(cfg.q4.extent_m), window=fc.window,
                                     gate=gate, outlier_iou=fc.outlier_iou,
                                     leak_frac=fc.leak_frac, band_px=fc.band_px,
                                     patience=fc.patience)
        grids, trace, times = [], [], []
        for b, s, p in zip(bevs, seen, gposes):
            t0 = time.perf_counter()
            g, st = filt.step(b, s, p)
            times.append(time.perf_counter() - t0)
            grids.append(g)
            trace.append(st)
        out[f"vote_{gate}"] = (grids, trace, times)
    return out


def plan_frame(cfg: Q4Config, specs, est: np.ndarray, start, goals, true_clr, true_safe) -> tuple[dict, dict]:
    """Build on `est`, run every planner to every goal, grade on the truth.
    Returns (roadmap stats, {tag: {key: [per goal]}})."""
    rm = build(cfg, est)
    res = {}
    for spec in specs:
        outs = [outcome(spec, rm, start, g, true_clr, true_safe, cfg.q4.robot_radius)
                for g in goals]
        res[tag(spec)] = {k: [o.get(k) for o in outs] for k in OUT_KEYS}
    stats = {"build_s": rm.build_s, "n_nodes": rm.n_nodes, "n_bridges": rm.n_bridges,
             "est_safe_frac": float(rm.safe.mean())}
    return stats, res


def eval_condition(cfg: Q4Config, f, base: dict, cond: dict, bevs, gt, specs, *,
                   img_stats=None, sam_s=None, erode_s=None, bev_s=None, is_bad=None) -> str:
    """Write the frame rows of one condition (all streams). Returns a
    one-line summary for the console."""
    summary = []
    for stream, (grids, trace, ftimes) in streams(cfg, bevs, gt["seen"], gt["gposes"]).items():
        n = c = 0
        for t, est in enumerate(grids):
            if gt["start"][t] is None or not gt["goals"][t]:
                continue
            stats, res = plan_frame(cfg, specs, est, gt["start"][t], gt["goals"][t],
                                    gt["clr"][t], gt["safe"][t])
            st = trace[t]
            row = {
                "row": "frame", **base, **cond, "stream": stream, "frame": t,
                "frame_id": int(gt["frame_ids"][t]),
                "frame_bad": None if is_bad is None else bool(is_bad[t]),
                "gate": None if st is None else st.gate,
                "gate_iou": None if st is None else st.iou,
                "gate_leak": None if st is None else st.leak,
                "n_vote": None if st is None else len(st.history),
                **(img_stats[t] if img_stats is not None else {}),
                **bev_stats(gt["occ"][t], est, gt["seen"][t], gt["sdf"][t]),
                **stats,
                "lat": {"sam_s": None if sam_s is None else sam_s[t],
                        "erode_s": None if erode_s is None else erode_s[t],
                        "bev_s": None if bev_s is None else bev_s[t],
                        "filter_s": ftimes[t], "build_s": stats["build_s"],
                        "plan_s": {k: v["plan_s"][0] for k, v in res.items()}},
                "start": list(gt["start"][t]), "goals": [list(g) for g in gt["goals"][t]],
                "out": res,
            }
            f.write(json.dumps(row) + "\n")
            for v in res.values():
                n += len(v["collision"])
                c += sum(v["collision"])
        summary.append(f"{stream} c{c}/{n}")
    f.flush()
    return " ".join(summary)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def sequence_truth(cfg: Q4Config, seq, n: int) -> dict:
    q4 = cfg.q4
    ext = tuple(q4.extent_m)
    robot = robot_cell(q4.px_per_m, ext)
    gt = {k: [] for k in ("mask", "occ", "seen", "clr", "safe", "sdf", "start", "goals")}
    for i in range(n):
        cam = seq.camera(i)
        m = seq.floor_masks[i]
        occ = mask_to_bev(m, seq.intrinsics, cam, q4.px_per_m, ext, q4.max_range_m, q4.supersample)
        clr, safe = truth(occ, q4.robot_radius)
        rng = np.random.default_rng([q4.seed, stable_seed(seq.name), i])
        start, goals = pick_queries(safe, robot, q4.goals_per_frame,
                                    q4.goal_dist_m[0] * q4.px_per_m,
                                    q4.goal_dist_m[1] * q4.px_per_m, rng)
        gt["mask"].append(m)
        gt["occ"].append(occ)
        gt["seen"].append(fov_mask(seq.intrinsics, cam, q4.px_per_m, ext, q4.max_range_m))
        gt["clr"].append(clr)
        gt["safe"].append(safe)
        gt["sdf"].append(corruption.signed_floor_distance(occ))
        gt["start"].append(start)
        gt["goals"].append(goals)
    gt["gposes"] = ground_poses(seq.poses[:n])
    gt["frame_ids"] = seq.frame_ids[:n]
    return gt


def to_bev(cfg: Q4Config, seq, masks) -> tuple[list[np.ndarray], list[float]]:
    q4 = cfg.q4
    out, times = [], []
    for i, m in enumerate(masks):
        t0 = time.perf_counter()
        out.append(mask_to_bev(m, seq.intrinsics, seq.camera(i), q4.px_per_m, tuple(q4.extent_m),
                               q4.max_range_m, q4.supersample))
        times.append(time.perf_counter() - t0)
    return out, times


def run(cfg: Q4Config, out_dir: Path, fake: bool = False, data_dir: str | None = None) -> Path:
    q4 = cfg.q4
    data = Path(data_dir or q4.data_dir)
    if not data.is_absolute():
        data = ROOT / data
    paths = list_sequences(data)
    if q4.sequences is not None:
        by_name = {p.name: p for p in paths}
        missing = [s for s in q4.sequences if s not in by_name]
        if missing:
            raise SystemExit(f"sequences not found under {data}: {missing}")
        paths = [by_name[s] for s in q4.sequences]
    if not paths:
        raise SystemExit(f"no sequences under {data}; see docs/Q4_DATA.md")
    cache_dir = out_dir / "masks" if fake else Path(q4.mask_cache_dir)
    if not cache_dir.is_absolute() and not fake:
        cache_dir = ROOT / cache_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    machine = machine_info()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    specs = planner_specs(cfg)
    t_all = time.perf_counter()
    warmup(cfg)
    with open(out_path, "w") as f:
        for si, path in enumerate(paths):
            t_seq = time.perf_counter()
            seq = load_sequence(path)
            n = len(seq) if q4.max_frames is None else min(len(seq), q4.max_frames)
            frames = list(seq.frames[:n])
            gt = sequence_truth(cfg, seq, n)
            h, w = frames[0].shape[:2]
            pf = [p for p in q4.prompt_frames if p < n] or [0]
            prompts = default_floor_prompts(h, w, frames=pf)
            base = {"started": started, "sequence": seq.name}
            f.write(json.dumps({
                "row": "sequence", **base, "n_frames": n,
                "camera_height_m": float(np.median(seq.camera_height_m[:n])),
                "pitch_deg_max": float(np.degrees(np.abs(seq.pitch_rad[:n]).max())),
                "floor_fit_spread_m": seq.floor_fit_spread_m,
                "path_m": float(np.hypot(*np.diff(gt["gposes"][:, :2], axis=0).T).sum()),
                "yaw_change_deg": float(np.degrees(np.abs(np.diff(np.unwrap(gt["gposes"][:, 2]))).sum())),
                "gt_free_frac": float(np.mean([(~o).mean() for o in gt["occ"]])),
                "seen_frac": float(np.mean([s.mean() for s in gt["seen"]])),
                "n_query_frames": int(sum(bool(g) for g in gt["goals"])),
                "n_goals": int(sum(len(g) for g in gt["goals"])),
                "prompts": json.loads(prompts.to_json()), "machine": machine,
                "fake_segmenter": fake,
            }) + "\n")

            if q4.control:
                bevs = gt["occ"]
                s = eval_condition(cfg, f, base, {"cond": "gt", "source": "gt"}, bevs, gt, specs)
                print(f"{seq.name:24s} gt: {s}")

            for bad in q4.inject.bad:
                name, level = bad
                rng = np.random.default_rng([q4.seed, stable_seed(seq.name, name, level)])
                is_bad = rng.random(n) < q4.inject.p_bad
                fn = corruption.CORRUPTIONS[name]
                bevs = [fn(o, level, rng) | ~s_ if b else o
                        for o, s_, b in zip(gt["occ"], gt["seen"], is_bad)]
                cond = {"cond": f"inject:{name}={level:g}", "source": "inject",
                        "inject": [name, level], "p_bad": q4.inject.p_bad}
                s = eval_condition(cfg, f, base, cond, bevs, gt, specs, is_bad=is_bad)
                print(f"{seq.name:24s} inject {name}={level:g}: {s}")

            for model in q4.models:
                for rule in q4.rules:
                    masks, info = run_sam(cfg, seq, frames, gt["mask"], model, rule, prompts,
                                          fake, cache_dir)
                    f.write(json.dumps({"row": "sam", **base, "model": model, "rule": rule,
                                        **info}) + "\n")
                    for e in q4.erosion_px:
                        er, er_s = [], []
                        for m in masks:
                            t0 = time.perf_counter()
                            er.append(erode(m, e))
                            er_s.append(time.perf_counter() - t0)
                        img = []
                        for m, g in zip(er, gt["mask"]):
                            st = corruption.mask_error_stats(~g, ~m)
                            img.append({"img_" + k: st[k] for k in IMG_KEYS}
                                       | {"img_floor_frac": float(m.mean())})
                        bevs, bev_s = to_bev(cfg, seq, er)
                        cond = {"cond": f"sam2.1-{model}/{rule}/e{e}", "source": "sam",
                                "model": model, "rule": rule, "erode_px": e}
                        s = eval_condition(cfg, f, base, cond, bevs, gt, specs, img_stats=img,
                                           sam_s=info["frame_s"], erode_s=er_s, bev_s=bev_s)
                        print(f"{seq.name:24s} {cond['cond']:38s} {s}")
            print(f"{seq.name}: {n} frames in {time.perf_counter() - t_seq:.1f} s")
            if q4.cooldown_s and si < len(paths) - 1:
                time.sleep(q4.cooldown_s)
    print(f"wrote {out_path} in {time.perf_counter() - t_all:.1f} s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override q4.out_dir")
    ap.add_argument("--data", help="override q4.data_dir")
    ap.add_argument("--fake-segmenter", action="store_true",
                    help="tests only: GT masks with a known error instead of SAM 2 (no weights)")
    args = ap.parse_args()
    cfg = load_q4_config(args.config)
    run(cfg, Path(args.out or cfg.q4.out_dir), fake=args.fake_segmenter, data_dir=args.data)


if __name__ == "__main__":
    main()
