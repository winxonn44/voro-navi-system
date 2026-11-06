"""Drive core_bench's BM_JfaKernel / BM_JfaQos for the C4 optimization log
(docs/CORE.md), one configuration per process with a cool-down in between.

Each selected benchmark runs as its own `core_bench` invocation with a short
warm-up, then `--reps` single-iteration repetitions, so every repetition is
one `run_steps` call and the median / min / max are over those calls. Rows
are appended to a JSONL file with the git commit, compiler, flags and power
state; the script refuses to run on battery power (design doc A6).

    python scripts/bench_core_kernels.py --suite h1 --out docs/runs/c4_h1.jsonl
    python scripts/bench_core_kernels.py --filter 'BM_JfaKernel/1/1/3/10/1/4/2048' ...

Build the benchmarks first:
    cmake -S core -B core/build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \\
        -DVORONAV_BUILD_TESTS=OFF
    cmake --build core/build-release --target core_bench
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "core" / "build-release" / "bench" / "core_bench"

KERNELS = {1: "reference", 2: "split", 3: "packed"}
VARIANTS = {1: "jfa+1", 2: "jfa+2"}
SCHEDULES = {0: "static", 1: "dynamic"}

# Name regexes per log entry (arguments: variant/density/kernel/threads/
# schedule/chunk/n; see core/bench/bench_jfa.cpp).
SUITES = {
    # H1 / H2: every kernel, serial and 10 threads dynamic, the A6 inputs.
    "kernels": r"^BM_JfaKernel/[12]/(1|10)/[123]/(1|10)/1/4/(1024|2048)/",
    # Same at 4 threads.
    "kernels4": r"^BM_JfaKernel/[12]/(1|10)/[123]/4/1/4/(1024|2048)/",
    # H3: chunk x schedule at 4 and 10 threads, packed, jfa+1 0.1%.
    "chunks": r"^BM_JfaKernel/1/1/3/(4|10)/[01]/(1|2|4|8|16|32)/(1024|2048)/",
    # Thread scaling at 2048^2, reference and packed, both schedules.
    "scaling": r"^BM_JfaKernel/1/1/[13]/(1|2|4|6|8|10)/[01]/4/2048/",
    # H5: worker QoS.
    "qos": r"^BM_JfaQos/",
    # Final table extra: 4096^2.
    "big": r"^BM_JfaKernel/[12]/(1|10)/3/(1|10)/1/4/4096/",
}


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def power_state() -> str:
    """'AC Power' or 'Battery Power' (macOS pmset), plus the charge line."""
    out = sh("pmset", "-g", "batt")
    if not out:
        return "unknown"
    first = re.search(r"'([^']+)'", out)
    charge = re.search(r"(\d+)%", out)
    src = first.group(1) if first else out.splitlines()[0]
    return f"{src} ({charge.group(1)}%)" if charge else src


def build_info(bench: Path) -> dict:
    # <build>/bench/core_bench, or a saved binary next to a copied cache.
    cache = next((d / "CMakeCache.txt" for d in (bench.parent, bench.parents[1])
                  if (d / "CMakeCache.txt").exists()), None)
    info: dict = {"binary": str(bench.relative_to(ROOT)) if bench.is_relative_to(ROOT)
                  else str(bench)}
    if cache is not None:
        for line in cache.read_text().splitlines():
            for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                        "CMAKE_CXX_FLAGS_RELEASE"):
                if line.startswith(key + ":"):
                    info[key] = line.split("=", 1)[1]
    if "CMAKE_CXX_COMPILER" in info:
        info["compiler"] = sh(info["CMAKE_CXX_COMPILER"], "--version").splitlines()[0:1]
    return info


def git_info() -> dict:
    return {
        "commit": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "branch": sh("git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain", "--", "core")),
        # Identifies uncommitted kernel code: sha1 of `git diff HEAD -- core/src core/include`.
        "diff_sha1": hashlib.sha1(
            sh("git", "-C", str(ROOT), "diff", "HEAD", "--", "core/src", "core/include").encode()
        ).hexdigest(),
    }


def parse_name(name: str) -> dict:
    parts = name.split("/")
    if parts[0] == "BM_JfaKernel":
        v, d, k, t, s, c, n = (int(p) for p in parts[1:8])
        return {"bench": "kernel", "variant": VARIANTS[v], "density": d / 1000,
                "kernel": KERNELS[k], "threads": t, "schedule": SCHEDULES[s] if t > 1 else None,
                "chunk": c if t > 1 and s == 1 else None, "size": n, "qos": "default"}
    if parts[0] == "BM_JfaQos":
        q, s, t, n = (int(p) for p in parts[1:5])
        return {"bench": "qos", "variant": "jfa+1", "density": 0.001, "kernel": "packed",
                "threads": t, "schedule": SCHEDULES[s], "chunk": 4 if s == 1 else None,
                "size": n, "qos": "user-interactive" if q else "default"}
    return {"bench": parts[0]}


def run_one(bench: Path, name: str, reps: int, warmup_s: float) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        subprocess.run(
            [str(bench), f"--benchmark_filter=^{re.escape(name)}$",
             f"--benchmark_repetitions={reps}", "--benchmark_min_time=1x",
             f"--benchmark_min_warmup_time={warmup_s}",
             "--benchmark_out_format=json", f"--benchmark_out={tmp.name}",
             "--benchmark_format=console"],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(Path(tmp.name).read_text())
    runs = [b for b in data["benchmarks"] if b.get("run_type") == "iteration"]
    times = [b["real_time"] for b in runs]  # ms (Unit kMillisecond)
    assert all(b["time_unit"] == "ms" for b in runs)
    return {
        "name": name,
        **parse_name(name),
        "passes": int(runs[0].get("passes", 0)),
        "times_ms": times,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "reps": len(times),
        "context": {k: data["context"].get(k) for k in
                    ("date", "num_cpus", "mhz_per_cpu", "library_build_type")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", action="append", default=[], choices=sorted(SUITES))
    ap.add_argument("--filter", action="append", default=[], help="extra name regex")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup-s", type=float, default=0.3)
    ap.add_argument("--cooldown-s", type=float, default=3.0)
    ap.add_argument("--label", default="", help="free-form tag stored in every row")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    patterns = [re.compile(SUITES[s]) for s in args.suite] + [re.compile(f) for f in args.filter]
    if not patterns:
        ap.error("give at least one --suite or --filter")
    listed = sh(str(args.bench), "--benchmark_list_tests").splitlines()
    names = [n for n in listed if any(p.search(n) for p in patterns)]
    if not names:
        ap.error("no benchmark matches")

    meta = {"git": git_info(), "build": build_info(args.bench), "label": args.label,
            "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "os": sh("sw_vers", "-productVersion")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} configurations -> {args.out}", flush=True)
    for i, name in enumerate(names):
        power = power_state()
        if "Battery" in power:
            print(f"on battery power ({power}); stopping (design doc A6)", file=sys.stderr)
            return 2
        if i:
            time.sleep(args.cooldown_s)
        row = run_one(args.bench, name, args.reps, args.warmup_s)
        row.update(meta)
        row["power"] = power
        row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(names)}] {name}: median {row['median_ms']:.2f} ms "
              f"(min {row['min_ms']:.2f}, max {row['max_ms']:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
