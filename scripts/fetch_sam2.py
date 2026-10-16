"""Fetch the SAM 2.1 checkpoints used by Q4 (tiny and small).

    python scripts/fetch_sam2.py              # download + verify both
    python scripts/fetch_sam2.py --model tiny # just one
    python scripts/fetch_sam2.py --dry-run    # list what would be downloaded

Files go to data/raw/sam2/ (gitignored). Each is checked against the byte size
and sha256 below; a mismatch is an error and the partial file is deleted.
Meta publishes no checksums, so the sha256 values were pinned on our first
download (2026-10-16) and guard against later corruption or upstream swaps.
The URLs are the ones in facebookresearch/sam2's checkpoints/download_ckpts.sh.
Checkpoints are Apache-2.0, like the code; see docs/Q4_SAM.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAM2_DIR = ROOT / "data" / "raw" / "sam2"
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# model -> (file name, size in bytes, sha256)
CHECKPOINTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", 156008466,
             "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"),
    "small": ("sam2.1_hiera_small.pt", 184416285,
              "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, size: int, digest: str | None) -> str | None:
    """Return an error message, or None if the file is good."""
    if path.stat().st_size != size:
        return f"{path.name}: {path.stat().st_size} bytes, expected {size}"
    got = file_sha256(path)
    if digest is None:
        print(f"  {path.name}: sha256 {got} (not pinned yet)")
    elif got != digest:
        return f"{path.name}: sha256 {got}, expected {digest}"
    return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False, suffix=".part"
    ) as tmp:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            tmp.write(chunk)
    Path(tmp.name).replace(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(CHECKPOINTS), action="append")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    errors = []
    for model in args.model or list(CHECKPOINTS):
        name, size, digest = CHECKPOINTS[model]
        dest = SAM2_DIR / name
        url = f"{BASE_URL}/{name}"
        if args.dry_run:
            state = "present" if dest.exists() else "missing"
            print(f"{model}: {url} ({size / 1e6:.1f} MB) -> {dest} [{state}]")
            continue
        if not dest.exists():
            download(url, dest)
        err = verify(dest, size, digest)
        if err:
            dest.unlink()
            errors.append(err)
        else:
            print(f"  {model}: ok ({dest.relative_to(ROOT)})")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
