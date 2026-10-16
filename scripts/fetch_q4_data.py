"""Fetch the Q4 first-person video subset (TartanGround, CC BY 4.0).

    python -I scripts/fetch_q4_data.py              # download + verify
    python -I scripts/fetch_q4_data.py --dry-run    # list what would be fetched
    python -I scripts/fetch_q4_data.py --probe      # print each zip's frame index
    python -I scripts/fetch_q4_data.py --select     # maintainer: fetch the
                                                    # windows in SELECTION and
                                                    # rewrite docs/q4_manifest.json

TartanGround ships one zip per trajectory per camera per modality (0.6-2.8 GB
for front RGB alone). We only need a few hundred frames, so the zips are never
downloaded whole: `RemoteFile` gives `zipfile` a seekable file whose reads are
HTTP Range requests, which fetches the central directory plus exactly the
members we ask for. Hugging Face's CDN supports ranges, and the URLs are
pinned to one dataset revision.

Output, one directory per sequence under data/raw/q4/ (gitignored):

    <seq>/image/<i>.png    front-left RGB, 640x640
    <seq>/seg/<i>.png      semantic label ids (uint8)
    <seq>/depth/<i>.png    float32 depth (m) packed into RGBA, every Nth frame
    <seq>/pose_lcam_front.txt, <seq>/metadata.json, <seq>/seg_label_map.json

Every written file is checked against the sha256 in docs/q4_manifest.json; a
mismatch is an error. Member names are mapped to fixed output paths, so a
hostile archive cannot write outside data/raw/q4/. See docs/Q4_DATA.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "q4"
MANIFEST = ROOT / "docs" / "q4_manifest.json"

REPO = "theairlabcmu/TartanGround"
REVISION = "388faf9c800568cfc6828fa47e063f8369397eb3"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
CAM = "lcam_front"
DEPTH_EVERY = 10  # depth is only used to fit the floor plane

# Maintainer selection: (sequence name, env, trajectory dir, first frame, n).
# Each window is the 150-frame stretch (start on a multiple of 25) with the
# most motion (path length + 2 x total yaw change) among windows where camera
# z varies < 5 cm and |pitch| < 1 deg, i.e. a flat floor; see docs/Q4_DATA.md.
SELECTION = [
    ("Hospital_P1000", "Hospital", "Data_diff/P1000", 2150, 150),
    ("AbandonedSchool_P1001", "AbandonedSchool", "Data_diff/P1001", 325, 150),
    ("Office_P0000", "Office", "Data_omni/P0000", 75, 150),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RemoteFile(io.RawIOBase):
    """Read-only seekable view of a URL over HTTP Range requests.

    `prefetch(start, end)` fetches one exact span (a zip member) in one
    request; other reads (central directory) go through BLOCK-sized,
    cached requests. Members are scattered through the archives, not stored
    in frame order, so whole blocks per member would waste most of the
    transfer."""

    BLOCK = 1 << 20
    CACHE = 4

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.bytes_read = 0
        self._blocks: dict[int, bytes] = {}
        self._span: tuple[int, bytes] = (0, b"")

    def prefetch(self, start: int, end: int) -> None:
        end = min(end, self.size)
        self._span = (start, self._get(start, end))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        rng = f"bytes={start}-{end - 1}"
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": rng})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if len(data) != end - start:
            raise OSError(f"short range read from {self.url}: {len(data)} != {end - start}")
        self.bytes_read += len(data)
        return data

    def _block(self, k: int) -> bytes:
        if k not in self._blocks:
            if len(self._blocks) >= self.CACHE:
                self._blocks.pop(next(iter(self._blocks)))
            start = k * self.BLOCK
            self._blocks[k] = self._get(start, min(self.size, start + self.BLOCK))
        return self._blocks[k]

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        s0, span = self._span
        if s0 <= self.pos and self.pos + n <= s0 + len(span):
            b[:n] = span[self.pos - s0 : self.pos - s0 + n]
        elif n > 2 * self.BLOCK:  # big read: one direct request
            b[:n] = self._get(self.pos, self.pos + n)
        else:
            done = 0
            while done < n:
                k, off = divmod(self.pos + done, self.BLOCK)
                chunk = self._block(k)[off : off + n - done]
                b[done : done + len(chunk)] = chunk
                done += len(chunk)
        self.pos += n
        return n


def open_zip(relpath: str) -> tuple[zipfile.ZipFile, RemoteFile]:
    raw = RemoteFile(f"{BASE}/{relpath}")
    return zipfile.ZipFile(raw), raw


def frame_members(zf: zipfile.ZipFile, suffix: str) -> dict[int, str]:
    """frame index -> member name for members like '.../000123_lcam_front_depth.png'."""
    out = {}
    for name in zf.namelist():
        m = re.search(r"(\d+)[^/]*" + re.escape(suffix) + r"$", name.rsplit("/", 1)[-1])
        if m and not name.endswith("/"):
            out[int(m.group(1))] = name
    return out


def zip_paths(env: str, traj: str) -> dict[str, str]:
    t = f"{env}/{traj}"
    return {
        "image": f"{t}/image_{CAM}.zip",
        "seg": f"{t}/seg_{CAM}.zip",
        "depth": f"{t}/depth_{CAM}.zip",
        "meta": f"{t}/metadata.zip",
        "labels": f"{env}/seg_labels.zip",
    }


def plan_sequence(name: str, env: str, traj: str, start: int, n: int) -> dict:
    """Open the remote zips and map each output file to (zip, member)."""
    files: dict[str, dict] = {}
    zips = zip_paths(env, traj)
    for kind in ("image", "seg", "depth"):
        zf, _ = open_zip(zips[kind])
        members = frame_members(zf, ".png")
        frames = range(start, start + n)
        if kind == "depth":
            frames = [i for i in frames if (i - start) % DEPTH_EVERY == 0]
        for i in frames:
            if i not in members:
                raise SystemExit(f"{zips[kind]} has no frame {i}")
            files[f"{name}/{kind}/{i:06d}.png"] = {"zip": zips[kind], "member": members[i]}
    zf, _ = open_zip(zips["meta"])
    for m in zf.namelist():
        base = m.rsplit("/", 1)[-1]
        if base == f"pose_{CAM}.txt":
            files[f"{name}/pose_{CAM}.txt"] = {"zip": zips["meta"], "member": m}
        elif base.endswith("metadata.json"):
            files[f"{name}/metadata.json"] = {"zip": zips["meta"], "member": m}
    zf, _ = open_zip(zips["labels"])
    for m in zf.namelist():
        if m.endswith("seg_label_map.json"):
            files[f"{name}/seg_label_map.json"] = {"zip": zips["labels"], "member": m}
    for required in (f"{name}/pose_{CAM}.txt", f"{name}/seg_label_map.json"):
        if required not in files:
            raise SystemExit(f"{name}: {required} not found in the archives")
    return files


def fetch_files(files: dict[str, dict], verify: bool) -> int:
    """Download the planned members, grouped by zip and read in archive
    order. Files already on disk are kept if they match the manifest (or,
    when selecting, if there is no hash yet). Returns bytes downloaded."""
    by_zip: dict[str, list[str]] = {}
    for out, spec in files.items():
        by_zip.setdefault(spec["zip"], []).append(out)
    total = 0
    for zpath, outs in by_zip.items():
        todo = []
        for out in outs:
            dest = OUT / out
            spec = files[out]
            if dest.exists():
                data = dest.read_bytes()
                if "sha256" not in spec or sha256(data) == spec["sha256"]:
                    spec["sha256"], spec["bytes"] = sha256(data), len(data)
                    continue
            todo.append(out)
        if not todo:
            continue
        print(f"  {zpath}: {len(todo)} members", flush=True)
        zf, raw = open_zip(zpath)
        for out in sorted(todo, key=lambda o: zf.getinfo(files[o]["member"]).header_offset):
            spec = files[out]
            info = zf.getinfo(spec["member"])
            # local header (30 B) + name + extra field; the local extra field
            # can differ from the central one, hence the slack
            head = 30 + len(info.orig_filename.encode()) + len(info.extra) + 1024
            raw.prefetch(info.header_offset, info.header_offset + head + info.compress_size)
            data = zf.read(spec["member"])
            if verify and sha256(data) != spec["sha256"]:
                raise SystemExit(f"sha256 mismatch for {out} ({zpath}:{spec['member']})")
            spec["sha256"] = sha256(data)
            spec["bytes"] = len(data)
            dest = OUT / out
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            total += len(data)
        print(f"    {raw.bytes_read / 1e6:.1f} MB transferred", flush=True)
    return total


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--select", action="store_true")
    args = ap.parse_args()

    if args.probe:
        for name, env, traj, _, _ in SELECTION:
            zf, raw = open_zip(zip_paths(env, traj)["image"])
            idx = sorted(frame_members(zf, ".png"))
            print(f"{name}: {len(idx)} frames ({idx[0]}..{idx[-1]}), "
                  f"zip {raw.size / 1e9:.2f} GB, index read {raw.bytes_read / 1e6:.2f} MB")
        return 0

    if args.select:
        seqs = []
        for name, env, traj, start, n in SELECTION:
            print(f"{name}: planning {n} frames from {start}")
            files = plan_sequence(name, env, traj, start, n)
            fetch_files(files, verify=False)
            seqs.append({"name": name, "env": env, "trajectory": traj, "camera": CAM,
                         "first_frame": start, "n_frames": n, "depth_every": DEPTH_EVERY,
                         "files": files})
        manifest = {
            "description": "Q4 first-person subset of TartanGround; written by "
                           "scripts/fetch_q4_data.py --select, verified by fetch_q4_data.py.",
            "source": f"https://huggingface.co/datasets/{REPO}",
            "revision": REVISION,
            "license": "CC BY 4.0",
            "sequences": seqs,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
        return 0

    manifest = load_manifest()
    if manifest["revision"] != REVISION:
        print("warning: manifest revision differs from the script's REVISION", file=sys.stderr)
    files = {k: v for s in manifest["sequences"] for k, v in s["files"].items()}
    size = sum(v["bytes"] for v in files.values())
    if args.dry_run:
        for s in manifest["sequences"]:
            sb = sum(v["bytes"] for v in s["files"].values())
            print(f"{s['name']}: {s['n_frames']} frames from {s['env']}/{s['trajectory']}, "
                  f"{len(s['files'])} files, {sb / 1e6:.0f} MB")
        print(f"total {size / 1e6:.0f} MB into {OUT.relative_to(ROOT)}/")
        return 0
    n = fetch_files(files, verify=True)
    print(f"ok: {len(files)} files verified ({n / 1e6:.0f} MB downloaded this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
