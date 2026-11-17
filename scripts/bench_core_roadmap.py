"""C5 timing: Python vs C++ `build_roadmap` on a handful of map sizes.

Build time only (JFA plus extraction); the end-to-end Q2 re-run is C6. Each
builder runs on the same map in the same session: one warm-up call, then
--reps timed calls (median and min/max are reported), with a cool-down between
configurations. Refuses to run on battery: on the fanless M4 Air the timings are
not worth recording otherwise.

    python scripts/bench_core_roadmap.py --out docs/runs/c5_roadmap.jsonl

Settings are bench_q2.yaml's (jfa+2, bridge_lam 64, robot radius 2). Rows record
the power state, compiler and the git commit (dirty hash of core/ if uncommitted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from voronav.config import load_q2_config
from voronav.map_sources import expand
from voronav.maps import random_obstacles
from voronav.planning.roadmap import build_roadmap

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC_SIZES = (256, 512, 1024, 2048)
REAL_MAPS = ("movingai_dir-room-", "movingai_dir-maze-", "houseexpo_dir-")


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def power_state() -> str:
    out = sh("pmset", "-g", "batt")
    first = re.search(r"'([^']+)'", out)
    charge = re.search(r"(\d+)%", out)
    src = first.group(1) if first else "unknown"
    return f"{src} ({charge.group(1)}%)" if charge else src


def require_ac() -> str:
    state = power_state()
    if "AC Power" not in state:
        raise SystemExit(f"refusing to measure on {state!r}: plug in the laptop and re-run")
    return state


def git_state() -> dict:
    diff = sh("git", "-C", str(REPO), "diff", "HEAD", "--", "core", "src/voronav/planning/roadmap.py")
    return {
        "commit": sh("git", "-C", str(REPO), "rev-parse", "--short", "HEAD"),
        "core_diff_sha1": hashlib.sha1(diff.encode()).hexdigest()[:12] if diff else None,
    }


def real_maps(cfg):
    """One map per real source, by name prefix (absent if the data is missing)."""
    out = {}
    for spec in cfg.q2.maps:
        if spec.kind not in ("movingai_dir", "houseexpo_dir"):
            continue
        root = REPO / spec.dir
        if not root.is_dir():
            continue
        from dataclasses import replace

        for m in expand(replace(spec, dir=str(root))):
            for prefix in REAL_MAPS:
                if m.name.startswith(prefix) and prefix not in out:
                    out[prefix] = m
    return list(out.values())


def time_builder(fn, reps: int) -> list[float]:
    fn()  # warm-up: Numba compile, first-touch pages
    out = []
    for _ in range(reps):
        out.append(fn())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--config", default=str(REPO / "configs" / "bench_q2.yaml"))
    args = ap.parse_args()

    import voronav_core as core

    cfg = load_q2_config(args.config)
    r = cfg.roadmap
    kw = dict(
        robot_radius=cfg.q2.robot_radius,
        min_angle_deg=r.min_angle_deg,
        raster_slack_px=r.raster_slack_px,
        min_piece_px=r.min_piece_px,
        bridge_lam=r.bridge_lam,
    )
    maps = [(f"random_obstacles-{s}-s0", random_obstacles(s, s, 25, seed=0)) for s in SYNTHETIC_SIZES]
    maps += [(m.name, m.occ) for m in real_maps(cfg)]

    common = {
        "git": git_state(),
        "settings": {"variant": r.variant, **kw},
        "numba_threads": "all (default)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"power: {require_ac()}; {len(maps)} maps, {args.reps} reps each")
    with open(args.out, "w") as f:
        for name, occ in maps:
            occ = np.ascontiguousarray(occ)
            py = lambda: build_roadmap(occ, backend=r.backend, variant=r.variant, **kw).build_s
            cpp = lambda: build_roadmap(occ, variant=r.variant, builder="cpp", **kw).build_s
            cpp1 = lambda: core.build_roadmap(occ, variant=r.variant, threads=1, **kw)["build_s"]
            rows = {}
            for label, fn in (("python-numba", py), ("cpp-allthreads", cpp), ("cpp-1thread", cpp1)):
                state = require_ac()
                times = time_builder(fn, args.reps)
                row = dict(
                    common,
                    map=name,
                    height=occ.shape[0],
                    width=occ.shape[1],
                    builder=label,
                    median_ms=statistics.median(times) * 1e3,
                    min_ms=min(times) * 1e3,
                    max_ms=max(times) * 1e3,
                    times_ms=[t * 1e3 for t in times],
                    reps=args.reps,
                    power=state,
                )
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows[label] = row["median_ms"]
                time.sleep(args.cooldown)
            n = int(build_roadmap(occ, variant=r.variant, builder="cpp", **kw).skeleton.sum())
            print(
                f"{name:42s} {occ.shape[0]:5d}x{occ.shape[1]:<5d} nodes={n:6d}  "
                f"py {rows['python-numba']:9.1f}  cpp {rows['cpp-allthreads']:8.1f}  "
                f"cpp1 {rows['cpp-1thread']:8.1f}  "
                f"{rows['python-numba'] / rows['cpp-allthreads']:5.2f}x"
            )


if __name__ == "__main__":
    main()
