"""Markdown tables for the C4 optimization log in docs/CORE.md,
from the JSONL files written by scripts/bench_core_kernels.py.

    python scripts/summarize_core_c4.py docs/runs/c4_h1_h2.jsonl [more.jsonl ...]

Prints one Markdown row per configuration: median (min–max) in ms, with the
power state recorded when it ran.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["file"] = p.name
                rows.append(r)
    return rows


def fmt(r: dict) -> str:
    return f"{r['median_ms']:.2f} ({r['min_ms']:.2f}–{r['max_ms']:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    rows = load(args.files)
    key = lambda r: (r["file"], r.get("size", 0), r.get("variant", ""), r.get("density", 0),  # noqa: E731
                     r.get("threads", 0), str(r.get("schedule")), r.get("chunk") or 0,
                     r.get("qos", ""), r.get("kernel", ""))
    print("| file | size | variant | density | kernel | threads | schedule | chunk | qos | "
          "median ms (min–max) | reps | power |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=key):
        print(f"| {r['file']} | {r.get('size')}² | {r.get('variant')} | {r.get('density'):.1%} | "
              f"{r.get('kernel')} | {r.get('threads')} | {r.get('schedule') or '-'} | "
              f"{r.get('chunk') or '-'} | {r.get('qos')} | {fmt(r)} | {r['reps']} | "
              f"{r.get('power')} |")


if __name__ == "__main__":
    main()
