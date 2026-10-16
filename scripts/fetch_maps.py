"""Fetch the real benchmark maps for the Q2 planner study.

    python scripts/fetch_maps.py              # download + extract + verify
    python scripts/fetch_maps.py --no-scen    # skip MovingAI .scen files
    python scripts/fetch_maps.py --dry-run    # list what would be downloaded
    python scripts/fetch_maps.py --select     # maintainer: re-pick the maps
                                              # from downloaded archives and
                                              # rewrite docs/maps_manifest.json

Archives go to data/raw/archives/, selected maps to
data/raw/movingai/<set>/<name>.map (with <name>.map.scen next to it) and
data/raw/houseexpo/<id>.json. Every extracted file is checked against the
sha256 in docs/maps_manifest.json; a mismatch is an error. An archive whose
own hash changed upstream is only a warning, since the files that matter are
checked one by one. See docs/DATA.md for sources, licenses and the selection.

Archives are read member by member (never extractall), so a hostile archive
cannot write outside data/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from voronav.maps import HOUSEEXPO_PX_PER_METER, load_houseexpo, load_movingai

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ARCHIVES = RAW / "archives"
MANIFEST = ROOT / "docs" / "maps_manifest.json"

MOVINGAI = "https://movingai.com/benchmarks"
HOUSEEXPO_COMMIT = "45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c"
HOUSEEXPO_URL = (
    f"https://raw.githubusercontent.com/TeaganLi/HouseExpo/{HOUSEEXPO_COMMIT}"
    "/HouseExpo/json.tar.gz"
)
MOVINGAI_SETS = ("dao", "bg512", "street", "maze", "room")
MAX_SIDE = 512
N_HOUSES = 20


def archive_urls() -> dict[str, tuple[str, bool]]:
    """archive file name -> (url, is_scen)."""
    out = {}
    for s in MOVINGAI_SETS:
        out[f"{s}-map.zip"] = (f"{MOVINGAI}/{s}/{s}-map.zip", False)
        out[f"{s}-scen.zip"] = (f"{MOVINGAI}/{s}/{s}-scen.zip", True)
    out["houseexpo-json.tar.gz"] = (HOUSEEXPO_URL, False)
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False
    ) as tmp:
        while chunk := r.read(1 << 20):
            tmp.write(chunk)
    Path(tmp.name).replace(dest)


def read_members(archive: Path, names: set[str] | None) -> dict[str, bytes]:
    """Bytes of the named regular-file members (one pass over the archive)."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            listed = set(z.namelist())
            return {n: z.read(n) for n in (listed if names is None else names & listed)}
    out = {}
    with tarfile.open(archive, "r:gz") as t:
        for m in t:
            if m.isfile() and (names is None or m.name in names):
                out[m.name] = t.extractfile(m).read()
    return out


def safe_dest(rel: str) -> Path:
    dest = (RAW / rel).resolve()
    if RAW.resolve() not in dest.parents:
        raise ValueError(f"manifest path escapes data/raw: {rel}")
    return dest


# ---------------------------------------------------------------------------
# Fetch + verify (default)
# ---------------------------------------------------------------------------


def fetch(manifest: dict, with_scen: bool, dry_run: bool) -> int:
    needed = {m["archive"] for m in manifest["maps"]}
    if with_scen:
        needed |= {m["scen"]["archive"] for m in manifest["maps"] if m.get("scen")}
    archives = manifest["archives"]

    print("archives:")
    for name in sorted(needed):
        a = archives[name]
        have = (ARCHIVES / name).exists()
        print(f"  {name:24s} {a['bytes'] / 1e6:6.2f} MB  {'(cached)' if have else a['url']}")
    if dry_run:
        return 0

    for name in sorted(needed):
        path, a = ARCHIVES / name, archives[name]
        if not path.exists():
            download(a["url"], path)
        if file_sha256(path) != a["sha256"]:
            print(f"  warning: {name} differs from the manifest (changed upstream?); "
                  "checking extracted files individually")

    items = []  # (archive, member, path, sha256)
    for m in manifest["maps"]:
        items.append((m["archive"], m["member"], m["path"], m["sha256"]))
        if with_scen and m.get("scen"):
            s = m["scen"]
            items.append((s["archive"], s["member"], s["path"], s["sha256"]))
    todo: dict[str, list] = {}
    for it in items:
        if not (safe_dest(it[2]).exists() and file_sha256(safe_dest(it[2])) == it[3]):
            todo.setdefault(it[0], []).append(it)

    errors = 0
    for archive, its in todo.items():
        data = read_members(ARCHIVES / archive, {it[1] for it in its})
        for _, member, rel, digest in its:
            if member not in data:
                print(f"  ERROR {rel}: {member} missing from {archive}")
                errors += 1
            elif sha256(data[member]) != digest:
                print(f"  ERROR {rel}: sha256 mismatch")
                errors += 1
            else:
                safe_dest(rel).parent.mkdir(parents=True, exist_ok=True)
                safe_dest(rel).write_bytes(data[member])

    for m in manifest["maps"]:
        if not safe_dest(m["path"]).exists():
            continue  # already reported
        occ = load_map(m)
        if occ.shape != (m["height"], m["width"]):
            print(f"  ERROR {m['path']}: size {occ.shape} != manifest "
                  f"{(m['height'], m['width'])}")
            errors += 1

    by_set: dict[str, list[dict]] = {}
    for m in manifest["maps"]:
        by_set.setdefault(m["set"], []).append(m)
    for s, ms in by_set.items():
        sides = sorted({f"{m['height']}x{m['width']}" for m in ms})
        print(f"  {s:10s} {len(ms):3d} maps  ({', '.join(sides[:4])}"
              f"{', ...' if len(sides) > 4 else ''})")
    print(f"{len(manifest['maps'])} maps in {RAW}, {errors} error(s)")
    return 1 if errors else 0


def load_map(m: dict) -> np.ndarray:
    path = safe_dest(m["path"])
    if m["set"] == "houseexpo":
        return load_houseexpo(path)
    return load_movingai(path)


# ---------------------------------------------------------------------------
# Selection (maintainer only; the manifest is the source of truth after)
# ---------------------------------------------------------------------------


def _movingai_header(data: bytes) -> tuple[int, int]:
    head = data[:200].decode()
    h = int(re.search(r"height (\d+)", head).group(1))
    w = int(re.search(r"width (\d+)", head).group(1))
    return h, w


def _evenly(items: list, k: int) -> list:
    """k items evenly spaced through a sorted list (deterministic, spread)."""
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def pick_movingai(s: str, members: list[str], free: dict[str, float]) -> list[str]:
    """10 maps per set; see docs/DATA.md for the reasoning."""
    if s == "maze":  # corridor widths 2..32 (width 1 has no clearance to speak of)
        return [f"maze512-{w}-{i}.map" for w in (2, 4, 8, 16, 32) for i in (0, 1)]
    if s == "room":  # room sizes 8..64 px, 512x512 each
        return [f"{r}room_00{i}.map" for r, n in ((8, 2), (16, 2), (32, 3), (64, 3))
                for i in range(n)]
    if s == "street":  # one 512 px map per city
        cities = sorted({m.split("_")[0] for m in members})
        return [f"{c}_0_512.map" for c in cities]
    # dao, bg512: evenly through the eligible maps (big enough, enough free space)
    return _evenly(sorted(m for m in members if free[m] >= 0.2), 10)


def select() -> dict:
    urls = archive_urls()
    missing = [n for n in urls if not (ARCHIVES / n).exists()]
    if missing:
        sys.exit(f"--select needs every archive in {ARCHIVES}; missing {missing}")
    archives = {
        n: {"url": u, "sha256": file_sha256(ARCHIVES / n),
            "bytes": (ARCHIVES / n).stat().st_size, "scen": is_scen}
        for n, (u, is_scen) in urls.items()
    }

    maps = []
    for s in MOVINGAI_SETS:
        with zipfile.ZipFile(ARCHIVES / f"{s}-map.zip") as z:
            members = sorted(n for n in z.namelist() if n.endswith(".map"))
            free, shapes = {}, {}
            for n in members:
                data = z.read(n)
                h, w = _movingai_header(data)
                if max(h, w) > MAX_SIDE or min(h, w) < 128:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".map") as tmp:
                    tmp.write(data)
                    tmp.flush()
                    occ = load_movingai(tmp.name)
                free[n], shapes[n] = float((~occ).mean()), occ.shape
            chosen = pick_movingai(s, [n for n in members if n in free], free)
            with zipfile.ZipFile(ARCHIVES / f"{s}-scen.zip") as zs:
                scen_names = set(zs.namelist())
                for n in chosen:
                    stem = n.removesuffix(".map")
                    scen = None
                    if f"{n}.scen" in scen_names:
                        scen = {"archive": f"{s}-scen.zip", "member": f"{n}.scen",
                                "path": f"movingai/{s}/{n}.scen",
                                "sha256": sha256(zs.read(f"{n}.scen"))}
                    maps.append({
                        "name": f"{s}-{stem}", "set": s, "source": "movingai",
                        "source_url": f"{MOVINGAI}/{s}/{s}-map.zip",
                        "archive": f"{s}-map.zip", "member": n,
                        "path": f"movingai/{s}/{n}", "sha256": sha256(z.read(n)),
                        "height": shapes[n][0], "width": shapes[n][1],
                        "free_fraction": round(free[n], 4), "scen": scen,
                    })

    # HouseExpo: shuffle the sorted ids with a fixed seed and take the first
    # N_HOUSES with 4-12 rooms that fit in MAX_SIDE at the default resolution.
    houses = read_members(ARCHIVES / "houseexpo-json.tar.gz", None)
    order = sorted(n for n in houses if n.endswith(".json"))
    np.random.default_rng(0).shuffle(order)
    picked = []
    for name in order:
        data = houses[name]
        doc = json.loads(data)
        if not 4 <= doc.get("room_num", 0) <= 12:
            continue
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            tmp.write(data)
            tmp.flush()
            occ = load_houseexpo(tmp.name)
        if max(occ.shape) > MAX_SIDE:
            continue
        hid = Path(name).stem
        picked.append({
            "name": f"houseexpo-{hid[:8]}", "set": "houseexpo", "source": "houseexpo",
            "source_url": HOUSEEXPO_URL, "archive": "houseexpo-json.tar.gz",
            "member": name, "path": f"houseexpo/{hid}.json", "sha256": sha256(data),
            "height": occ.shape[0], "width": occ.shape[1],
            "free_fraction": round(float((~occ).mean()), 4),
            "room_num": doc["room_num"], "scen": None,
        })
        if len(picked) == N_HOUSES:
            break
    maps += sorted(picked, key=lambda m: m["name"])

    return {
        "description": "Real maps for the Q2 planner study; written by "
                       "scripts/fetch_maps.py --select, verified by fetch_maps.py.",
        "houseexpo_commit": HOUSEEXPO_COMMIT,
        "houseexpo_px_per_meter": HOUSEEXPO_PX_PER_METER,
        "archives": archives,
        "maps": maps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-scen", action="store_true", help="skip MovingAI .scen files")
    ap.add_argument("--dry-run", action="store_true", help="list archives, fetch nothing")
    ap.add_argument("--select", action="store_true",
                    help="re-pick maps from downloaded archives, rewrite the manifest")
    args = ap.parse_args()
    if args.select:
        manifest = select()
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(manifest['maps'])} maps)")
        return 0
    return fetch(json.loads(MANIFEST.read_text()), not args.no_scen, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
