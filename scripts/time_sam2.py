"""Per-frame latency of SAM 2.1 tiny / small video segmentation, MPS vs CPU.

    python scripts/time_sam2.py --frames DIR             # a folder of JPEG/PNG frames
    python scripts/time_sam2.py --frames DIR -n 50 --models tiny small --devices mps cpu
    python scripts/time_sam2.py --frames DIR --devices mps --precision fp32 bf16

For each (model, device, precision) it runs Sam2Floor's video path once on the first n
frames with the default floor prompts, and reports:
  setup   model build + checkpoint load (once per process and model)
  first   prompt frame: frame loading and encoding (init_state), the multimask
          decode and selection, and add_new_mask; this is the warmup cost
  frame   every propagated frame after the prompt frame: median, p90, mean
Precision "auto" is Sam2Floor's default (bf16 on MPS, fp32 on CPU); half
precision is skipped on CPU.  Raw numbers go to runs/q4_sam_timing.json.  One short run; the M4 Air is
fanless, so expect later configs to run a little warmer than the first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from voronav.vision.segment import Sam2Floor, _quiet, _swap, default_floor_prompts

ROOT = Path(__file__).resolve().parent.parent


def load_frames(folder: Path, n: int) -> list[np.ndarray]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths[:n]]


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_one(model: str, device: str, precision: str | None, frames: list[np.ndarray]) -> dict:
    import sam2.sam2_video_predictor as svp

    t0 = time.perf_counter()
    seg = Sam2Floor(model, device=device, precision=precision)
    sync(device)
    setup = time.perf_counter() - t0

    h, w = frames[0].shape[:2]
    prompts = default_floor_prompts(h, w)
    per_frame, fg = [], []
    with (torch.inference_mode(), seg._autocast(), _quiet(),
          _swap(svp, "tqdm", lambda it, **_: it)):
        t0 = time.perf_counter()
        state = seg._init_state(frames)
        masks, k = seg._candidates(frames[0], prompts)
        seg.predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=masks[k])
        sync(device)
        t_prev = time.perf_counter()
        first = t_prev - t0
        for idx, _, logits in seg.predictor.propagate_in_video(state, start_frame_idx=0):
            fg.append(float((logits[0, 0] > 0).float().mean().cpu()))  # .cpu() syncs
            t = time.perf_counter()
            if idx == 0:  # the prompt frame: already decoded, only consolidated
                first += t - t_prev
            else:
                per_frame.append(t - t_prev)
            t_prev = t
        seg.predictor.reset_state(state)
    a = np.asarray(per_frame)
    return {
        "model": model, "device": device, "precision": seg.precision, "n_frames": len(frames), "hw": [h, w],
        "setup_s": setup, "first_s": first,
        "frame_median_s": float(np.median(a)), "frame_p90_s": float(np.percentile(a, 90)),
        "frame_mean_s": float(a.mean()), "frames_s": per_frame,
        "floor_fraction": fg, "choice": seg.last_choices,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["tiny", "small"])
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--precision", nargs="+", default=["auto"],
                    choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "q4_sam_timing.json")
    ap.add_argument("--source", default=None, help="free-text note on where the frames came from")
    args = ap.parse_args(argv)

    frames = load_frames(args.frames, args.n)
    print(f"{len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]} from {args.frames}")
    rows = []
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps not available, skipped")
            continue
        for model, prec in [(m, p) for m in args.models for p in args.precision]:
            if prec not in ("auto", "fp32") and device != "mps":
                continue
            r = time_one(model, device, None if prec == "auto" else prec, frames)
            rows.append(r)
            print(f"{model:5s} {device:3s} {r['precision']}  setup {r['setup_s']:.2f} s  first {r['first_s']:.2f} s  "
                  f"frame median {r['frame_median_s'] * 1e3:.0f} ms  p90 {r['frame_p90_s'] * 1e3:.0f} ms  "
                  f"({1 / r['frame_median_s']:.1f} fps)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"torch": torch.__version__, "source": args.source or str(args.frames),
            "date": time.strftime("%Y-%m-%d %H:%M")}
    args.out.write_text(json.dumps({"meta": meta, "runs": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
