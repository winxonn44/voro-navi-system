"""Q4 first-person sequences with ground-truth floor masks (TartanGround).

Layout of one sequence directory (written by scripts/fetch_q4_data.py):

    <seq>/image/<frame>.png      RGB, 640x640, 90 deg FOV pinhole
    <seq>/seg/<frame>.png        uint8 semantic ids, see seg_label_map.json
    <seq>/depth/<frame>.png      planar depth (m), float32 packed in RGBA;
                                 only every Nth frame, used to fit the floor
    <seq>/pose_lcam_front.txt    one row per trajectory frame:
                                 x y z qx qy qz qw, NED world, camera body
                                 frame x forward / y right / z down
    <seq>/seg_label_map.json     {"name_map": {class name: id}}
    <seq>/metadata.json          includes robot_height (m)

Frame numbers are trajectory indices, so frame f uses pose row f.

Camera placement over the floor, per frame:
  pitch, roll  from the pose quaternion (exact in simulation);
  height       floor_z - camera z, where floor_z is ONE floor plane per
               sequence, fitted as the median world z of ground-truth floor
               pixels back-projected through the depth frames. Sequences were
               picked on flat floors (camera z varies < 5 cm), so one plane
               is right; `floor_fit_spread_m` reports how flat it really is.
"""

from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .bev import CameraPose, Intrinsics

CAM = "lcam_front"
HFOV_RAD = np.pi / 2  # TartanAir / TartanGround pinhole cameras: 90 deg
# Class names that count as walkable floor. Exact matches against the
# per-environment seg_label_map.json; stairs, ramps and floor-standing
# objects ("floorstand", "floorradiator") are deliberately not floor. The
# decals ("floordirt", painted court lines) lie on the floor plane (checked
# against depth: 99.7% of floordirt pixels within 2%). Environments that
# label floor and walls alike as "building" (Supermarket, Restaurant) can't
# be used: no class name separates their floor.
FLOOR_NAMES = frozenset(
    {"floor", "ground", "carpet", "rug", "floorrug", "floormat", "floordirt",
     "floorbasketballmarkings", "groundbasketballmarking"}
)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_seg(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel label image, got {a.shape}")
    return a


def read_depth(path: Path) -> np.ndarray:
    """TartanAir float32 depth: the 4 little-endian bytes of each value are
    the pixel's channels in OpenCV's BGRA order (tartanairpy reads it with
    cv2.imread + view('<f4')), so swap PIL's RGBA to BGRA first."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]]).view("<f4")[..., 0]


def write_depth(path: Path, depth: np.ndarray) -> None:
    """Inverse of read_depth (used to build test fixtures)."""
    a = np.ascontiguousarray(depth, dtype="<f4")[..., None].view(np.uint8)
    Image.fromarray(np.ascontiguousarray(a[..., [2, 1, 0, 3]]), mode="RGBA").save(path)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[..., 4] (qx, qy, qz, qw) -> [..., 3, 3] rotation (body -> world)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def pitch_roll_from_ned(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pitch (positive = looking down) and roll (positive = right side down)
    of a body->NED rotation with body x forward, y right, z down."""
    pitch = np.arcsin(np.clip(R[..., 2, 0], -1, 1))
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return pitch, roll


class _LazyImages(SequenceABC):
    """Read-on-index list of images; nothing is cached."""

    def __init__(self, paths: list[Path], reader):
        self._paths = paths
        self._reader = reader

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._reader(p) for p in self._paths[i]]
        return self._reader(self._paths[i])


@dataclass
class Sequence:
    name: str
    root: Path
    frame_ids: np.ndarray  # int [N], trajectory indices
    intrinsics: Intrinsics
    camera_height_m: np.ndarray  # float [N]
    pitch_rad: np.ndarray  # float [N]
    roll_rad: np.ndarray  # float [N]
    poses: np.ndarray  # float [N, 7] x y z qx qy qz qw (NED)
    floor_ids: tuple[int, ...]
    floor_z_ned: float
    floor_fit_spread_m: float  # MAD of floor-point z around the fitted plane
    frames: _LazyImages = field(repr=False)
    floor_masks: _LazyImages = field(repr=False)
    depth_frame_ids: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def camera(self, i: int) -> CameraPose:
        return CameraPose(float(self.camera_height_m[i]), float(self.pitch_rad[i]), float(self.roll_rad[i]))

    def depth(self, frame_id: int) -> np.ndarray:
        return read_depth(self.root / "depth" / f"{frame_id:06d}.png")


def floor_ids_from_label_map(path: Path, names=FLOOR_NAMES) -> tuple[int, ...]:
    name_map = json.loads(Path(path).read_text())["name_map"]
    return tuple(sorted(int(v) for k, v in name_map.items() if k.lower() in names))


def _frame_files(d: Path) -> dict[int, Path]:
    return {int(p.stem): p for p in d.glob("*.png") if p.stem.isdigit()} if d.is_dir() else {}


def fit_floor_z(
    seg_dir: Path,
    depth_dir: Path,
    poses_all: np.ndarray,
    intr: Intrinsics,
    floor_ids: tuple[int, ...],
    max_depth_m: float = 10.0,
) -> tuple[float, float]:
    """Median NED z of GT floor pixels with depth, and its MAD."""
    ids = np.array(floor_ids)
    zs = []
    vv, uu = np.mgrid[0 : intr.height, 0 : intr.width]
    for f, dpath in sorted(_frame_files(depth_dir).items()):
        spath = seg_dir / dpath.name
        if not spath.exists():
            continue
        floor = np.isin(read_seg(spath), ids)
        d = read_depth(dpath)
        ok = floor & np.isfinite(d) & (d > 0) & (d < max_depth_m)
        if ok.sum() < 50:
            continue
        z = d[ok]
        # optical (right, down, fwd) -> body (fwd, right, down)
        body = np.stack([z, (uu[ok] - intr.cx) / intr.fx * z, (vv[ok] - intr.cy) / intr.fy * z], -1)
        R = quat_to_matrix(poses_all[f, 3:])
        zs.append(body @ R[2] + poses_all[f, 2])
    if not zs:
        raise ValueError(f"{depth_dir}: no depth frame with enough floor pixels")
    z = np.concatenate(zs)
    med = float(np.median(z))
    return med, float(np.median(np.abs(z - med)))


def load_sequence(path: str | Path) -> Sequence:
    root = Path(path)
    poses_all = np.loadtxt(root / f"pose_{CAM}.txt", ndmin=2)
    if poses_all.shape[1] != 7:
        raise ValueError(f"{root}: pose rows must be x y z qx qy qz qw")
    images = _frame_files(root / "image")
    segs = _frame_files(root / "seg")
    ids = sorted(set(images) & set(segs))
    if not ids:
        raise ValueError(f"{root}: no frames with both image/ and seg/")
    if ids[-1] >= len(poses_all):
        raise ValueError(f"{root}: frame {ids[-1]} has no pose row")
    with Image.open(images[ids[0]]) as im:
        w, h = im.size
    intr = Intrinsics.from_fov(w, h, HFOV_RAD)
    floor_ids = floor_ids_from_label_map(root / "seg_label_map.json")
    if not floor_ids:
        raise ValueError(f"{root}: no floor class among {sorted(FLOOR_NAMES)}")

    poses = poses_all[ids]
    pitch, roll = pitch_roll_from_ned(quat_to_matrix(poses[:, 3:]))
    floor_z, spread = fit_floor_z(root / "seg", root / "depth", poses_all, intr, floor_ids)
    height = floor_z - poses[:, 2]  # NED: z grows downward
    if np.any(height <= 0):
        raise ValueError(f"{root}: fitted floor is above the camera")
    ids_arr = np.array(floor_ids)
    return Sequence(
        name=root.name,
        root=root,
        frame_ids=np.array(ids),
        intrinsics=intr,
        camera_height_m=height,
        pitch_rad=pitch,
        roll_rad=roll,
        poses=poses,
        floor_ids=floor_ids,
        floor_z_ned=floor_z,
        floor_fit_spread_m=spread,
        frames=_LazyImages([images[i] for i in ids], read_rgb),
        floor_masks=_LazyImages([segs[i] for i in ids], lambda p: np.isin(read_seg(p), ids_arr)),
        depth_frame_ids=np.array(sorted(_frame_files(root / "depth"))),
    )


def list_sequences(root: str | Path) -> list[Path]:
    """Sequence directories under root (those with poses and a label map)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if (p / f"pose_{CAM}.txt").exists() and (p / "seg_label_map.json").exists()
    )
