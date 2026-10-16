"""Why was the eager PyTorch MPS backend slow? Measure each suspect separately.

    python scripts/probe_mps.py --out docs/runs/q1_mps_probe.json

Hypotheses from the first Q1 run:
  H1  int64 arithmetic is slow or emulated on Apple GPUs (Metal is 32-bit native)
  H2  each pass allocates ~9 full-grid temporaries per candidate offset

Probes:
  1. single elementwise ops (add, floor-div, ...) on a 2048^2 grid in int64,
     int32 and float32, writing into a preallocated `out=` vs allocating;
  2. how many aten ops one eager pass dispatches;
  3. the full jfa+1 diagram at 2048^2 through six implementations that
     change one thing at a time: dtype of the eager ops, torch.compile fusion,
     and a hand-written fused Metal kernel with int64 or int32 storage/math.
Every implementation is checked against the Numba backend with array_equal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from voronav.maps import random_sites
from voronav.voronoi import init_nearest, step_schedule, voronoi
from voronav.voronoi import jfa_torch

DEV = "mps"

_HDR = "#include <metal_stdlib>\nusing namespace metal;\n"
_ARGS = """(device const {T}* cur [[buffer(0)]], device {T}* out [[buffer(1)]],
    constant long& k [[buffer(2)]], constant long& h [[buffer(3)]],
    constant long& w [[buffer(4)]], uint idx [[thread_position_in_grid]])"""

# Fused, int64 storage AND int64 math: the packed key, 64-bit `/` and `%`.
METAL_I64 = _HDR + "kernel void p" + _ARGS.format(T="long") + """ {
    long y = long(idx) / w; long x = long(idx) - y * w; long hw = h * w;
    long best = 0x7FFFFFFFFFFFFFFF;
    for (long a = -1; a <= 1; ++a) { long yy = y + a * k; if (yy < 0 || yy >= h) continue;
      for (long b = -1; b <= 1; ++b) { long xx = x + b * k; if (xx < 0 || xx >= w) continue;
        long s = cur[yy * w + xx]; if (s < 0) continue;
        long sy = s / w; long sx = s - sy * w;
        long d2 = (y - sy) * (y - sy) + (x - sx) * (x - sx);
        long key = d2 * hw + s; if (key < best) best = key; } }
    out[idx] = best == 0x7FFFFFFFFFFFFFFF ? -1 : best % hw;
}"""

# Fused, int64 storage, packed int64 key, but 32-bit coordinate division:
# one 64-bit `/` per pixel instead of ten.
METAL_I64_ONE_DIV = _HDR + "kernel void p" + _ARGS.format(T="long") + """ {
    int W = int(w), H = int(h), K = int(k); int y = int(idx) / W; int x = int(idx) - y * W;
    long hw = h * w; long best = 0x7FFFFFFFFFFFFFFF;
    for (int a = -1; a <= 1; ++a) { int yy = y + a * K; if (yy < 0 || yy >= H) continue;
      for (int b = -1; b <= 1; ++b) { int xx = x + b * K; if (xx < 0 || xx >= W) continue;
        long s = cur[yy * W + xx]; if (s < 0) continue;
        int si = int(s); int sy = si / W; int sx = si - sy * W;
        long d2 = long((y - sy) * (y - sy) + (x - sx) * (x - sx));
        long key = d2 * hw + s; if (key < best) best = key; } }
    out[idx] = best == 0x7FFFFFFFFFFFFFFF ? -1 : best - (best / hw) * hw;
}"""

# Fused, int64 storage, all-int32 math (lexicographic d2-then-site compare).
METAL_I64_STORE_I32_MATH = _HDR + "kernel void p" + _ARGS.format(T="long") + """ {
    int W = int(w), H = int(h), K = int(k); int y = int(idx) / W; int x = int(idx) - y * W;
    uint bd = 0xFFFFFFFFu; int bs = -1;
    for (int a = -1; a <= 1; ++a) { int yy = y + a * K; if (yy < 0 || yy >= H) continue;
      for (int b = -1; b <= 1; ++b) { int xx = x + b * K; if (xx < 0 || xx >= W) continue;
        int s = int(cur[yy * W + xx]); if (s < 0) continue;
        int sy = s / W; int sx = s - sy * W; int dy = y - sy, dx = x - sx;
        uint d2 = uint(dy * dy) + uint(dx * dx);
        if (d2 < bd || (d2 == bd && s < bs)) { bd = d2; bs = s; } } }
    out[idx] = long(bs);
}"""


def sync_time(fn, warmup: int, repeats: int) -> float:
    """Median wall time in ms, synchronizing the GPU around each run."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def probe_ops(n: int) -> dict:
    res = {}
    for dt in (torch.int64, torch.int32, torch.float32):
        if dt.is_floating_point:
            a = torch.rand(n, n, device=DEV) + 1
        else:
            a = torch.randint(1, 1000, (n, n), dtype=dt, device=DEV)
        b = a + 1
        o = torch.empty_like(a)
        r = {
            "add_out": sync_time(lambda: torch.add(a, b, out=o), 3, 20),
            "add_alloc": sync_time(lambda: a + b, 3, 20),
            "mul_out": sync_time(lambda: torch.mul(a, b, out=o), 3, 20),
            "minimum_out": sync_time(lambda: torch.minimum(a, b, out=o), 3, 20),
            "floordiv_out": sync_time(
                lambda: torch.div(a, b, rounding_mode="floor", out=o), 3, 20),
            "remainder_out": sync_time(lambda: torch.remainder(a, b, out=o), 3, 20),
        }
        res[str(dt).removeprefix("torch.")] = r
    return res


def count_ops(cur: torch.Tensor) -> int:
    from torch.profiler import ProfilerActivity, profile

    h, w = cur.shape
    py, px = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV),
                            indexing="ij")
    with profile(activities=[ProfilerActivity.CPU]) as p:
        jfa_torch.jfa_pass(cur, h // 4, py, px)
        torch.mps.synchronize()
    # Top-level ops only: every aten call that launches (or may launch) a kernel.
    names = ("add", "sub", "mul", "pow", "div", "remainder", "minimum", "where",
             "ge", "eq", "fill_", "copy_")
    return sum(e.count for e in p.key_averages() if e.key.removeprefix("aten::") in names)


def _shift(cur, dy, dx, fill):
    h, w = cur.shape
    cand = torch.full_like(cur, fill)
    if abs(dy) >= h or abs(dx) >= w:
        return cand
    cand[max(0, -dy): h - max(0, dy), max(0, -dx): w - max(0, dx)] = \
        cur[max(0, dy): h + min(0, dy), max(0, dx): w + min(0, dx)]
    return cand


def eager_int32_pass(cur, k, py, px):
    """The eager pass with int32 tensors; packed int64 keys would overflow,
    so it compares (d2, site) lexicographically instead."""
    _, w = cur.shape
    big = 2**31 - 1
    bd = torch.full_like(cur, big)
    bs = torch.full_like(cur, -1)
    for dy in (-k, 0, k):
        for dx in (-k, 0, k):
            cand = _shift(cur, dy, dx, -1)
            sy = torch.div(cand, w, rounding_mode="floor")
            sx = cand - sy * w
            d2 = torch.where(cand >= 0, (py - sy) ** 2 + (px - sx) ** 2, big)
            better = (d2 < bd) | ((d2 == bd) & (cand >= 0) & (cand < bs))
            bd = torch.where(better, d2, bd)
            bs = torch.where(better, cand, bs)
    return bs


def implementations():
    def eager(dtype, pass_fn):
        def run(init, steps):
            h, w = init.shape
            py, px = torch.meshgrid(torch.arange(h, dtype=dtype, device=DEV),
                                    torch.arange(w, dtype=dtype, device=DEV), indexing="ij")
            cur = init.to(dtype)
            for k in steps:
                cur = pass_fn(cur, k, py, px)
            return cur.to(torch.int64)
        return run

    def metal(src, dtype):
        lib = torch.mps.compile_shader(src)

        def run(init, steps):
            h, w = init.shape
            cur = init.to(dtype, copy=True).contiguous()  # never write into init
            nxt = torch.empty_like(cur)
            for k in steps:
                lib.p(cur, nxt, k, h, w, threads=h * w)
                cur, nxt = nxt, cur
            return cur.to(torch.int64)
        return run

    import torch._dynamo

    torch._dynamo.config.recompile_limit = 64  # one graph per step length k
    compiled = torch.compile(jfa_torch.jfa_pass, dynamic=False)
    return {
        "eager int64 (old backend)": eager(torch.int64, jfa_torch.jfa_pass),
        "eager int32": eager(torch.int32, eager_int32_pass),
        "torch.compile int64": eager(torch.int64, compiled),
        "metal fused, int64 storage + int64 math": metal(METAL_I64, torch.int64),
        "metal fused, int64 storage, one int64 div/pixel": metal(METAL_I64_ONE_DIV, torch.int64),
        "metal fused, int64 storage, int32 math": metal(METAL_I64_STORE_I32_MATH, torch.int64),
        "metal fused, int32 (new backend)":
            lambda init, steps: jfa_torch.run_steps(init, steps, fused=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--density", type=float, default=0.001)
    ap.add_argument("--variant", default="jfa+1")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("docs/runs/q1_mps_probe.json"))
    args = ap.parse_args()
    n = args.size

    sites = random_sites(n, n, round(args.density * n * n), seed=0)
    init = jfa_torch.to_device(init_nearest(sites), DEV)
    steps = step_schedule(n, n, args.variant)
    ref = voronoi(sites, backend="numba", variant=args.variant)

    out = {"size": n, "density": args.density, "variant": args.variant,
           "n_passes": len(steps), "torch": torch.__version__}
    out["elementwise_ms"] = probe_ops(n)
    out["eager_ops_per_pass"] = count_ops(init)
    print("elementwise ms:", json.dumps(out["elementwise_ms"], indent=1))
    print("eager aten ops per pass:", out["eager_ops_per_pass"])

    out["diagram_ms"] = {}
    for name, fn in implementations().items():
        equal = bool(np.array_equal(fn(init, steps).cpu().numpy(), ref))
        ms = sync_time(lambda: fn(init, steps), 2, args.repeats)
        out["diagram_ms"][name] = {"median_ms": ms, "bit_identical": equal}
        print(f"{name:50s} {ms:9.1f} ms  bit-identical={equal}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
