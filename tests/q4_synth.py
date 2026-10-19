"""A synthetic TartanGround-style sequence for Q4 tests: a level camera
driving forward (and turning a little) down a 4 m wide corridor with a box
on the floor. Labels come from the flat-floor geometry (no occlusion), so the
GT BEV is exact up to resampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from voronav.vision.bev import CameraPose, Intrinsics, image_to_ground
from voronav.vision.dataset import write_depth

FLOOR, WALL = 157, 236
FLOOR_Z = 1.0  # NED: floor 1 m below the world origin
BOX = (2.5, 3.2, -0.2, 0.4)  # world (x0, x1, y0, y1) m: north 2.5-3.2, east -0.2-0.4


def quat_yaw(yaw: float) -> np.ndarray:
    """NED body->world quaternion (qx, qy, qz, qw) for a level camera."""
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def render(size: int, height: float, x: float, y: float, yaw: float):
    intr = Intrinsics.from_fov(size, size, np.pi / 2)
    vv, uu = np.mgrid[0:size, 0:size].astype(float)
    g = image_to_ground(intr, CameraPose(height), np.stack([uu, vv], -1))  # X fwd, Y left
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([np.sin(yaw), -np.cos(yaw)])
    wx = x + g[..., 0] * f[0] + g[..., 1] * left[0]
    wy = y + g[..., 0] * f[1] + g[..., 1] * left[1]
    hit = np.isfinite(g[..., 0]) & (g[..., 0] < 30)
    floor = hit & (np.abs(wy) < 2.0)
    floor &= ~((wx > BOX[0]) & (wx < BOX[1]) & (wy > BOX[2]) & (wy < BOX[3]))
    seg = np.where(floor, FLOOR, WALL).astype(np.uint8)
    depth = np.where(hit, g[..., 0], 100.0).astype(np.float32)  # level camera: depth = X
    return seg, depth


def make_sequence(root: Path, name: str = "Synth_P0000", n: int = 8, size: int = 96,
                  height: float = 0.6, step_m: float = 0.1, yaw_step: float = 0.02) -> Path:
    d = Path(root) / name
    for sub in ("image", "seg", "depth"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    poses = []
    for i in range(n):
        yaw = yaw_step * i
        x, y = step_m * i, 0.05 * i
        poses.append([x, y, FLOOR_Z - height, *quat_yaw(yaw)])
        seg, depth = render(size, height, x, y, yaw)
        Image.fromarray(seg, mode="L").save(d / "seg" / f"{i:06d}.png")
        rgb = np.stack([seg, seg // 2, 255 - seg], -1).astype(np.uint8)
        Image.fromarray(rgb).save(d / "image" / f"{i:06d}.png")
        if i % 2 == 0:
            write_depth(d / "depth" / f"{i:06d}.png", depth)
    np.savetxt(d / "pose_lcam_front.txt", np.array(poses))
    (d / "seg_label_map.json").write_text(json.dumps({"name_map": {"floor": FLOOR, "wall": WALL}}))
    (d / "metadata.json").write_text(json.dumps({"robot_height": height}))
    return d
