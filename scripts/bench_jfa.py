"""Q1 benchmark harness: time every JFA backend across grid sizes, site
densities, variants and thread counts, and measure error against the exact
Euclidean distance transform. One JSON line per configuration.

    python scripts/bench_jfa.py --config configs/bench_q1.yaml

Error is backend-independent (the backends are bit-identical, see
tests/test_backends_agree.py), so it is computed once per
(size, density, variant) and copied onto each timing row.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from voronav.config import BenchConfig, load_config, to_dict
from voronav.maps import random_sites
from voronav.voronoi import distance_sq, make_runner, step_schedule


def machine_info() -> dict:
    import numba

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
    }
    try:
        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip() or platform.processor()
    except OSError:
        info["cpu"] = platform.processor()
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def exact_d2(sites: np.ndarray) -> np.ndarray:
    return np.rint(distance_transform_edt(~sites) ** 2).astype(np.int64)


def error_stats(out: np.ndarray, true_d2: np.ndarray) -> dict:
    d2 = distance_sq(out)
    wrong = d2 != true_d2
    excess = np.sqrt(d2[wrong]) - np.sqrt(true_d2[wrong]) if wrong.any() else np.zeros(1)
    return {
        "error_rate": float(wrong.mean()),
        "error_pixels": int(wrong.sum()),
        "max_excess_px": float(excess.max()),
    }


def time_runner(run, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return times


def torch_path(backend: str, device: str, size: int) -> str | None:
    """Which torch implementation ran: the fused Metal kernel or eager ops."""
    if backend != "torch":
        return None
    from voronav.voronoi.jfa_torch import fused_supported

    return "metal-int32" if fused_supported(size, size, device) else "eager-int64"


def configurations(cfg: BenchConfig):
    """(backend, device, threads, schedule) tuples to time."""
    from voronav.voronoi.jfa_numba import max_threads

    for backend in cfg.backends:
        if backend == "numpy":
            yield "numpy", "cpu", 1, None
        elif backend == "numba":
            for sch in cfg.numba_schedules:
                for t in cfg.numba_threads:
                    if t > max_threads():
                        print(f"  skip numba threads={t} (> {max_threads()} cores)", file=sys.stderr)
                        continue
                    yield "numba", "cpu", t, sch
        elif backend == "torch":
            from voronav.voronoi.jfa_torch import available_devices

            for d in cfg.torch_devices:
                if d not in available_devices():
                    print(f"  skip torch device={d} (unavailable)", file=sys.stderr)
                    continue
                yield "torch", d, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override bench.out_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = cfg.bench
    out_dir = Path(args.out or bench.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    machine = machine_info()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    combos = list(configurations(bench))

    with open(out_path, "w") as f:
        for size in bench.sizes:
            for density in bench.site_densities:
                n_sites = max(1, round(density * size * size))
                sites = random_sites(size, size, n_sites, seed=bench.seed)
                true_d2 = exact_d2(sites) if bench.measure_error else None
                for variant in bench.variants:
                    err = None
                    for backend, device, threads, schedule in combos:
                        run, fetch = make_runner(
                            sites, backend, variant, device, threads, schedule or "dynamic"
                        )
                        times = time_runner(run, bench.warmup, bench.repeats)
                        if true_d2 is not None and err is None:
                            err = error_stats(fetch(), true_d2)
                        row = {
                            "started": started,
                            "size": size,
                            "pixels": size * size,
                            "n_sites": n_sites,
                            "density": density,
                            "variant": variant,
                            "n_passes": len(step_schedule(size, size, variant)),
                            "backend": backend,
                            "device": device,
                            "threads": threads,
                            "schedule": schedule,
                            "torch_path": torch_path(backend, device, size),
                            "times_s": times,
                            "median_s": statistics.median(times),
                            "min_s": min(times),
                            "mpix_per_s": size * size / statistics.median(times) / 1e6,
                            **(err or {}),
                            "machine": machine,
                        }
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        tag = f"{backend}-{device}" + (
                            f"-{schedule[:3]}-t{threads}" if backend == "numba" else ""
                        )
                        print(
                            f"size={size:5d} sites={n_sites:6d} {variant:6s} {tag:18s} "
                            f"median={row['median_s'] * 1e3:9.2f} ms  "
                            f"err={row.get('error_rate', float('nan')):.2e}"
                        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
