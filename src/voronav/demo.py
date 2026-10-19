"""Offline demo: first-person video -> SAM 2 floor mask -> bird's-eye grid ->
Voronoi roadmap -> paths, rendered as an animated GIF.

    voronav demo data/raw/q4/Hospital_P1000 --out docs/q4/demo.gif
    voronav demo my_frames/ --fov 69 --height 0.8 --pitch 5 --out demo.gif
    python scripts/demo.py ...            (same, without installing the entry point)

Input is either a TartanGround sequence directory (voronav.vision.dataset:
camera height and pitch per frame, GT floor masks for an optional outline,
poses for the goal) or a directory of image frames (png/jpg, sorted by name)
plus the camera's horizontal field of view, height and pitch.

Each GIF frame has two panels: the camera image with SAM 2's floor mask
(and the ground-truth outline when available), and the bird's-eye grid
(dark = obstacle or unseen) with the Voronoi skeleton, the Voronoi best64
path (blue), the grid A* path (orange), the goal and the measured per-frame
latency of each stage. Start = the safe cell nearest the robot (the robot's
own cell is below the image, so unseen). Goal: for a TartanGround sequence,
where the robot actually is `--lookahead` frames later (from the poses),
snapped to the nearest reachable safe cell; otherwise, the reachable safe
cell furthest ahead. Everything runs offline; SAM 2 masks are cached in
runs/q4_masks like the Q4 harness. An MP4 is also written when ffmpeg is
already installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, label

from .planning import astar_plan, build_roadmap, voronoi_plan
from .planning.grid_search import REACHABILITY
from .vision.bev import CameraPose, Intrinsics, fov_mask, mask_to_bev, robot_cell
from .vision.dataset import list_sequences, load_sequence, read_rgb
from .vision.egomotion import ground_poses, relative_transform
from .vision.segment import CACHE_DIR, CachedSegmenter, default_floor_prompts

PPM, EXT, RANGE = 20.0, (5.0, 8.0), 5.0
RADIUS = 2.0
CELL = 3  # screen pixels per BEV cell
COL = {
    "bg": (252, 252, 251), "ink": (43, 43, 41), "muted": (138, 137, 127),
    "obstacle": (58, 58, 56), "unseen": (92, 92, 88), "free": (236, 235, 228),
    "skeleton": (143, 183, 232), "voronoi": (42, 120, 214), "astar": (235, 104, 52),
    "goal": (214, 69, 93), "robot": (27, 175, 122), "floor": (27, 175, 122), "gt": (255, 255, 255),
}


class _LazySam2:
    """Builds Sam2Floor only on a cache miss; records per-frame wall time."""

    def __init__(self, model: str):
        self.model = model
        self.last_frame_s = None

    @property
    def cache_tag(self) -> str:
        from .vision.segment import default_device
        prec = "bf16" if default_device() == "mps" else "fp32"
        return f"sam2.1-{self.model}/largest_consistent/erode0/{prec}"

    def segment_video(self, frames, prompts):
        from .vision.segment import Sam2Floor
        seg = Sam2Floor(model=self.model, select="largest_consistent")
        masks = seg.segment_video(frames, prompts)
        self.last_frame_s = seg.last_frame_s
        return masks

    def segment_image(self, frame, prompts):
        raise NotImplementedError


def segment(frames, name: str, model: str, cache_dir: Path) -> tuple[list[np.ndarray], list[float | None]]:
    h, w = frames[0].shape[:2]
    prompts = default_floor_prompts(h, w)
    inner = _LazySam2(model)
    seg = CachedSegmenter(inner, cache_dir)
    side = seg.path(frames, prompts, name).with_suffix(".info.json")
    masks = seg.segment_video(frames, prompts, sequence=name)
    if seg.misses:
        side.write_text(json.dumps({"frame_s": inner.last_frame_s}))
        times = inner.last_frame_s
    elif side.exists():
        times = json.loads(side.read_text())["frame_s"]
    else:
        times = [None] * len(frames)
    return masks, times


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_input(args) -> dict:
    src = Path(args.input)
    seqs = list_sequences(src.parent) if src.is_dir() else []
    if (src / "pose_lcam_front.txt").exists() and src in seqs:
        seq = load_sequence(src)
        idx = list(range(args.first, min(len(seq), args.first + args.frames * args.step), args.step))
        poses = ground_poses(seq.poses)
        return {"name": f"{seq.name}-{args.first}-{args.step}-{len(idx)}",
                "frames": [seq.frames[i] for i in idx],
                "gt": [seq.floor_masks[i] for i in idx] if args.gt_outline else None,
                "intr": seq.intrinsics, "cams": [seq.camera(i) for i in idx],
                "poses": poses, "idx": idx, "title": seq.name}
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"{src}: neither a TartanGround sequence nor a directory of frames")
    if args.height is None:
        raise SystemExit("a directory of frames needs --height (and --fov, --pitch)")
    files = files[args.first:args.first + args.frames * args.step:args.step]
    frames = [read_rgb(p) for p in files]
    h, w = frames[0].shape[:2]
    intr = Intrinsics.from_fov(w, h, np.radians(args.fov))
    cam = CameraPose(args.height, np.radians(args.pitch))
    return {"name": f"{src.name}-{args.first}-{args.step}-{len(files)}", "frames": frames,
            "gt": None, "intr": intr, "cams": [cam] * len(frames), "poses": None,
            "idx": None, "title": src.name}


# ---------------------------------------------------------------------------
# Planning on one frame
# ---------------------------------------------------------------------------


def pick_goal(safe: np.ndarray, start, target_rc=None):
    regions, _ = label(safe, structure=REACHABILITY)
    pix = np.argwhere(regions == regions[start])
    if len(pix) < 2:
        return None
    d_start = np.hypot(*(pix - np.array(start)).T)
    pix = pix[d_start >= 20]  # at least 1 m away
    if len(pix) == 0:
        return None
    if target_rc is not None:
        k = np.argmin(np.hypot(*(pix - np.asarray(target_rc)).T))
    else:
        h, w = safe.shape
        k = np.argmax(-pix[:, 0] - 0.5 * np.abs(pix[:, 1] - w / 2))  # far ahead, near the centre
    return int(pix[k][0]), int(pix[k][1])


def future_cell(poses, i: int, j: int):
    """Where the robot is at frame j, in frame i's BEV cell coordinates."""
    A, b = relative_transform(poses[i], poses[j])
    X, Y = b  # frame j's origin in frame i's ground coords
    return (EXT[0] - X) * PPM - 0.5, (EXT[1] / 2 - Y) * PPM - 0.5


def plan_frame(occ: np.ndarray, target_rc=None) -> dict:
    t0 = time.perf_counter()
    rm = build_roadmap(occ, robot_radius=RADIUS, bridge_lam=64)
    out = {"build_s": time.perf_counter() - t0, "rm": rm, "start": None, "goal": None,
           "vor": None, "astar": None, "plan_s": 0.0}
    cells = np.argwhere(rm.safe)
    if len(cells) == 0:
        return out
    robot = robot_cell(PPM, EXT)
    s = cells[np.argmin(np.hypot(*(cells - np.array(robot)).T))]
    start = (int(s[0]), int(s[1]))
    goal = pick_goal(rm.safe, start, target_rc)
    out["start"], out["goal"] = start, goal
    if goal is None:
        return out
    v = voronoi_plan(rm, start, goal, join="best", join_lam=64, roadmap_lam=64)
    a = astar_plan(rm.safe, start, goal)
    out["vor"] = v.path if v.success else None
    out["astar"] = a.path if a.success else None
    out["plan_s"] = v.plan_s
    out["astar_s"] = a.plan_s
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def camera_panel(rgb, mask, gt, size: int) -> Image.Image:
    img = rgb.astype(np.float32)
    tint = np.array(COL["floor"], np.float32)
    img[mask] = 0.55 * img[mask] + 0.45 * tint
    rim = binary_dilation(mask, iterations=2) & ~mask
    img[rim] = tint
    if gt is not None:
        g_rim = binary_dilation(gt, iterations=1) & ~gt
        img[g_rim] = COL["gt"]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)


def bev_panel(occ, seen, plan) -> Image.Image:
    h, w = occ.shape
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = COL["free"]
    rgb[occ & seen] = COL["obstacle"]
    rgb[occ & ~seen] = COL["unseen"]
    rm = plan["rm"]
    rgb[rm.skeleton] = COL["skeleton"]
    im = Image.fromarray(rgb).resize((w * CELL, h * CELL), Image.NEAREST)
    d = ImageDraw.Draw(im)
    pt = lambda rc: ((rc[1] + 0.5) * CELL, (rc[0] + 0.5) * CELL)
    for key, width in (("astar", 2), ("vor", 3)):
        p = plan.get(key)
        if p is not None and len(p) > 1:
            d.line([pt(q) for q in p], fill=COL["voronoi" if key == "vor" else "astar"], width=width)
    rr, rc = robot_cell(PPM, EXT)
    x, y = pt((rr, rc))
    d.polygon([(x, y - 9), (x - 6, y + 3), (x + 6, y + 3)], fill=COL["robot"])
    if plan["start"] is not None:
        sx, sy = pt(plan["start"])
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], outline=COL["robot"], width=2)
    if plan["goal"] is not None:
        gx, gy = pt(plan["goal"])
        d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=COL["goal"], width=3)
        d.line([gx - 3, gy, gx + 3, gy], fill=COL["goal"], width=2)
    return im


def compose(cam: Image.Image, bev: Image.Image, header: str, lat: str) -> Image.Image:
    pad, top, bottom = 10, 30, 52
    W = cam.width + bev.width + 3 * pad
    H = max(cam.height, bev.height) + top + bottom
    out = Image.new("RGB", (W, H), COL["bg"])
    out.paste(cam, (pad, top))
    out.paste(bev, (2 * pad + cam.width, top + max(0, cam.height - bev.height) // 2))
    d = ImageDraw.Draw(out)
    d.text((pad, 8), header, fill=COL["ink"], font=font(14))
    y = top + max(cam.height, bev.height) + 6
    d.text((pad, y), lat, fill=COL["ink"], font=font(13))
    x, y = pad, y + 20
    items = [("Voronoi best64 path", COL["voronoi"]), ("grid A* path", COL["astar"]),
             ("Voronoi skeleton", COL["skeleton"]), ("goal", COL["goal"]), ("robot", COL["robot"]),
             ("obstacle (seen)", COL["obstacle"]), ("unseen", COL["unseen"]), ("free", COL["free"])]
    for name, c in items:
        d.rectangle([x, y + 3, x + 14, y + 11], fill=c, outline=COL["muted"])
        d.text((x + 18, y), name, fill=COL["muted"], font=font(12))
        x += 30 + int(d.textlength(name, font=font(12)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="voronav demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="TartanGround sequence dir, or a directory of frames")
    ap.add_argument("--out", default="docs/q4/demo.gif")
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--frames", type=int, default=75, help="number of frames in the GIF")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--step", type=int, default=2, help="use every k-th frame")
    ap.add_argument("--fov", type=float, default=90.0, help="frames dir: horizontal FOV, degrees")
    ap.add_argument("--height", type=float, help="frames dir: camera height over the floor, m")
    ap.add_argument("--pitch", type=float, default=0.0, help="frames dir: pitch down, degrees")
    ap.add_argument("--lookahead", type=int, default=25, help="goal = robot position this many source frames later")
    ap.add_argument("--gt-outline", action="store_true", help="draw the ground-truth floor outline (white)")
    ap.add_argument("--size", type=int, default=300, help="camera panel size, px")
    ap.add_argument("--ms-per-frame", type=int, default=150)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = ap.parse_args(argv)

    data = load_input(args)
    frames = data["frames"]
    t0 = time.perf_counter()
    masks, sam_s = segment(frames, data["name"], args.model, Path(args.cache_dir))
    print(f"SAM 2.1 {args.model}: {len(frames)} frames in {time.perf_counter() - t0:.1f} s")

    images = []
    for i, (rgb, m) in enumerate(zip(frames, masks)):
        cam = data["cams"][i]
        t = time.perf_counter()
        occ = mask_to_bev(m, data["intr"], cam, PPM, EXT, RANGE)
        bev_s = time.perf_counter() - t
        seen = fov_mask(data["intr"], cam, PPM, EXT, RANGE)
        target = None
        if data["poses"] is not None:
            src = data["idx"][i]
            j = min(src + args.lookahead, len(data["poses"]) - 1)
            target = future_cell(data["poses"], src, j) if j > src else None
        plan = plan_frame(occ, target)
        parts = [("SAM", sam_s[i]), ("BEV", bev_s), ("roadmap", plan["build_s"]), ("plan", plan["plan_s"])]
        total = sum(v for _, v in parts if v is not None)
        fmt = lambda v: f"{1000 * v:.0f}" if v >= 0.01 else f"{1000 * v:.1f}"
        lat = "   ".join(f"{k} {fmt(v)} ms" if v is not None else f"{k} (cached)" for k, v in parts)
        lat += f"   =  {1000 * total:.0f} ms/frame on an M4 MacBook Air"
        if i == 0:
            lat += "  (first frame: model setup + prompt)"
        header = (f"{data['title']}  frame {data['idx'][i] if data['idx'] else i}   SAM 2.1 {args.model} "
                  f"floor mask (green){', ground truth outline (white)' if data['gt'] else ''}  |  bird's-eye grid, 5 cm cells")
        img = compose(camera_panel(rgb, m, None if data["gt"] is None else data["gt"][i], args.size),
                      bev_panel(occ, seen, plan), header, lat)
        images.append(img)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=args.ms_per_frame, loop=0,
                optimize=True, disposal=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(images)} frames)")
    if shutil.which("ffmpeg"):
        mp4 = out.with_suffix(".mp4")
        with tempfile.TemporaryDirectory() as tmp:
            for k, im in enumerate(images):
                im.save(Path(tmp) / f"{k:04d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(1000 / args.ms_per_frame),
                            "-i", str(Path(tmp) / "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            "-pix_fmt", "yuv420p", str(mp4)], check=True)
        print(f"wrote {mp4}")
    else:
        print("ffmpeg not found: GIF only")


if __name__ == "__main__":
    main()
