"""Flame graph (SVG) and folded stacks from a samply profile, for the C4
optimization log (docs/CORE.md). No browser or Firefox Profiler needed.

Record (macOS, no root needed for a launched child):

    samply record --save-only --unstable-presymbolicate -o prof.json.gz -- \\
        core/build-release/bench/core_bench --benchmark_filter='^BM_JfaKernel/1/1/3/1/1/4/2048/' \\
        --benchmark_min_time=2s

then

    python scripts/flamegraph_samply.py prof.json.gz --svg docs/flamegraphs/h4b.svg \\
        --title "H4b packed+SIMD, serial, 2048^2" [--folded out.folded] [--top 15]

`--unstable-presymbolicate` writes prof.json.syms.json next to the profile;
frame addresses are resolved against it (symbol tables only: Release builds
have no line info, and inlined functions appear as their caller). Only
samples whose stack contains `--root` (default: the benchmark function) are
kept, so process start-up and grid seeding are left out.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import html
import json
import re
from pathlib import Path


def load(path: Path):
    prof = json.load(gzip.open(path) if path.suffix == ".gz" else path.open())
    syms_path = Path(re.sub(r"\.gz$", "", str(path)) + "").with_suffix(".json.syms.json")
    if not syms_path.exists():
        syms_path = Path(str(path).replace(".json.gz", ".json.syms.json"))
    syms = json.load(syms_path.open())
    table = syms["string_table"]
    by_code = {}
    for lib in syms["data"]:
        entries = sorted(lib["symbol_table"], key=lambda e: e["rva"])
        by_code[lib["code_id"].upper()] = ([e["rva"] for e in entries], entries)
    return prof, table, by_code


def demangle_short(name: str) -> str:
    # Keep it readable: drop argument lists and voronav:: / anonymous namespace noise.
    name = name.replace("voronav::(anonymous namespace)::", "").replace("voronav::", "")
    name = name.replace("(anonymous namespace)::", "").replace("std::__1::", "std::")
    # Cut the argument list: the first '(' outside template brackets.
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0 and i > 0:
            return name[:i]
    return name


def folded(prof, table, by_code, root: str) -> collections.Counter:
    """Folded stacks under the first frame matching any '|'-separated root."""
    roots = root.split("|")
    libs = prof["libs"]
    out: collections.Counter = collections.Counter()
    for t in prof["threads"]:
        st, ft, fn, rt = t["stackTable"], t["frameTable"], t["funcTable"], t["resourceTable"]
        cache: dict[int, str] = {}

        def frame_name(fi: int) -> str:
            if fi in cache:
                return cache[fi]
            func = ft["func"][fi]
            res = fn["resource"][func]
            addr = ft["address"][fi]
            name = f"0x{addr:x}"
            if res is not None and res >= 0:
                lib = libs[rt["lib"][res]]
                key = (lib.get("codeId") or "").upper()
                if key in by_code:
                    rvas, entries = by_code[key]
                    i = bisect.bisect_right(rvas, addr) - 1
                    if i >= 0 and addr < entries[i]["rva"] + entries[i]["size"]:
                        name = demangle_short(table[entries[i]["symbol"]])
                    else:
                        name = f"{lib['name']}+0x{addr:x}"
            cache[fi] = name
            return name

        for si, w in zip(t["samples"]["stack"], t["samples"].get("weight") or
                         [1] * len(t["samples"]["stack"])):
            names = []
            while si is not None:
                names.append(frame_name(st["frame"][si]))
                si = st["prefix"][si]
            names.reverse()
            hit = next((i for i, n in enumerate(names) if any(r in n for r in roots)), None)
            if hit is None:
                continue
            out[";".join(names[hit:])] += w or 1
    return out


def svg(stacks: collections.Counter, title: str, width: int = 1200) -> str:
    # Build a tree, then lay it out root at the bottom (classic flame graph).
    tree: dict = {"n": 0, "c": {}}
    for stack, n in stacks.items():
        node = tree
        node["n"] += n
        for f in stack.split(";"):
            node = node["c"].setdefault(f, {"n": 0, "c": {}})
            node["n"] += n
    total = tree["n"] or 1

    def depth(node):
        return 1 + max((depth(c) for c in node["c"].values()), default=0)

    rows = depth(tree) - 1
    fh, top = 17, 40
    height = top + rows * fh + 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="Menlo,monospace" font-size="11">',
             '<rect width="100%" height="100%" fill="#fdfdf6"/>',
             f'<text x="10" y="20" font-size="14">{html.escape(title)}</text>',
             f'<text x="10" y="34" fill="#555">{total} samples; width = share of samples '
             'under the benchmark loop</text>']
    scale = (width - 20) / total

    def color(name: str) -> str:
        h = sum(map(ord, name)) % 40
        return f"rgb({205 + h // 2},{90 + h * 2},{50 + h})"

    def draw(node, name, x, level):
        w = node["n"] * scale
        if level >= 0 and w >= 0.5:
            y = height - 30 - (level + 1) * fh
            pct = 100 * node["n"] / total
            label = f"{name} ({pct:.1f}%)"
            parts.append(f'<g><title>{html.escape(label)}</title>'
                         f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{fh - 1}" '
                         f'fill="{color(name)}" rx="2"/>')
            chars = int(w / 6.6)
            if chars >= 4:
                text = label if len(label) <= chars else label[:chars - 2] + ".."
                parts.append(f'<text x="{x + 3:.1f}" y="{y + 12}">{html.escape(text)}</text>')
            parts.append("</g>")
        cx = x
        for cname, child in sorted(node["c"].items()):
            draw(child, cname, cx, level + 1)
            cx += child["n"] * scale

    draw(tree, "all", 10, -1)
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("--root", default="run_steps|worker_loop",
                    help="frames to cut stacks at ('|'-separated); the pool's workers start at worker_loop")
    ap.add_argument("--svg", type=Path)
    ap.add_argument("--folded", type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=12, help="print the top self-time frames")
    args = ap.parse_args()
    prof, table, by_code = load(args.profile)
    stacks = folded(prof, table, by_code, args.root)
    total = sum(stacks.values())
    leaf = collections.Counter()
    for s, n in stacks.items():
        leaf[s.rsplit(";", 1)[-1]] += n
    print(f"{total} samples under {args.root!r}; self time:")
    for name, n in leaf.most_common(args.top):
        print(f"  {100 * n / total:5.1f}%  {name}")
    if args.folded:
        args.folded.write_text("".join(f"{s} {n}\n" for s, n in sorted(stacks.items())))
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(svg(stacks, args.title or args.profile.name))
        print(f"wrote {args.svg}")


if __name__ == "__main__":
    main()
