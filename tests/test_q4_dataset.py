import json

import numpy as np
import pytest
from PIL import Image

from voronav.vision.bev import CameraPose, Intrinsics, image_to_ground, mask_to_bev
from voronav.vision.dataset import (
    floor_ids_from_label_map,
    list_sequences,
    load_sequence,
    pitch_roll_from_ned,
    quat_to_matrix,
    read_depth,
    write_depth,
)

W = H = 48
FLOOR, WALL, RUG = 157, 236, 33
FLOOR_Z = 1.4  # NED: floor 1.4 m below the world origin


def quat_yaw_pitch(yaw, pitch):
    """NED body->world quaternion (qx, qy, qz, qw) for yaw then pitch (nose down > 0)."""
    # ZYX: q = qz(yaw) * qy(-pitch); body pitch-up is +y rotation in NED,
    # so nose-down is a negative y rotation.
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    cy, sy = np.cos(-pitch / 2), np.sin(-pitch / 2)
    return np.array([-sz * sy, cz * sy, sz * cy, cz * cy])


def render(height, pitch):
    """Seg + planar depth for a camera over an infinite floor with a rug strip."""
    cam = CameraPose(height, pitch)
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    intr = Intrinsics.from_fov(W, H, np.pi / 2)
    xy = image_to_ground(intr, cam, np.stack([uu, vv], -1))
    hit = np.isfinite(xy[..., 0]) & (xy[..., 0] < 30)
    seg = np.full((H, W), WALL, np.uint8)
    seg[hit] = FLOOR
    seg[hit & (np.abs(xy[..., 1]) < 0.3)] = RUG
    # planar depth of a floor point: its coordinate along the optical axis
    fwd = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    pts = np.stack([xy[..., 0], xy[..., 1], np.zeros_like(xy[..., 0])], -1) - np.array([0, 0, height])
    depth = np.where(hit, pts @ fwd, 100.0).astype(np.float32)
    return seg, depth


@pytest.fixture
def seq_dir(tmp_path):
    root = tmp_path / "q4"
    d = root / "Office_P0000"
    for sub in ("image", "seg", "depth"):
        (d / sub).mkdir(parents=True)
    n_traj, frames = 10, [3, 4, 5, 6]
    heights = np.linspace(0.8, 0.85, n_traj)
    pitches = np.radians(np.linspace(0, 6, n_traj))
    poses = []
    for i in range(n_traj):
        q = quat_yaw_pitch(0.1 * i, pitches[i])
        poses.append([0.5 * i, -2.0, FLOOR_Z - heights[i], *q])
    np.savetxt(d / "pose_lcam_front.txt", np.array(poses))
    for f in frames:
        seg, depth = render(heights[f], pitches[f])
        Image.fromarray(seg, mode="L").save(d / "seg" / f"{f:06d}.png")
        rgb = np.stack([seg, seg // 2, 255 - seg], -1).astype(np.uint8)
        Image.fromarray(rgb).save(d / "image" / f"{f:06d}.png")
        if f % 2 == 1:
            write_depth(d / "depth" / f"{f:06d}.png", depth)
    (d / "seg_label_map.json").write_text(
        json.dumps({"name_map": {"floor": FLOOR, "wall": WALL, "floorrug": RUG, "floorstand": 7}})
    )
    (d / "metadata.json").write_text(json.dumps({"robot_height": 0.8}))
    (root / "not_a_sequence").mkdir()
    return root, d, heights, pitches, frames


def test_depth_roundtrip(tmp_path):
    d = np.random.default_rng(0).uniform(0.1, 50, (7, 5)).astype(np.float32)
    write_depth(tmp_path / "d.png", d)
    assert np.array_equal(read_depth(tmp_path / "d.png"), d)


def test_depth_byte_order(tmp_path):
    # 1.0f is bytes 00 00 80 3F (little-endian), stored as B, G, R, A, so a
    # PIL RGBA pixel of (0x80, 0, 0, 0x3F) decodes to 1.0.
    px = np.array([[[0x80, 0, 0, 0x3F]]], dtype=np.uint8)
    Image.fromarray(px, mode="RGBA").save(tmp_path / "one.png")
    assert read_depth(tmp_path / "one.png")[0, 0] == 1.0


def test_quaternion_pitch_roll():
    for yaw, pitch in [(0.0, 0.0), (1.0, 0.2), (-2.5, -0.1)]:
        R = quat_to_matrix(quat_yaw_pitch(yaw, pitch))
        assert np.allclose(R @ R.T, np.eye(3))
        p, r = pitch_roll_from_ned(R)
        assert np.isclose(p, pitch) and np.isclose(r, 0.0, atol=1e-9)
        # nose-down means the forward axis points toward +z (down) in NED
        assert np.sign(R[2, 0]) == np.sign(pitch) or pitch == 0


def test_floor_ids(seq_dir):
    _, d, *_ = seq_dir
    assert floor_ids_from_label_map(d / "seg_label_map.json") == (RUG, FLOOR)


def test_list_sequences(seq_dir, tmp_path):
    root, d, *_ = seq_dir
    assert list_sequences(root) == [d]
    assert list_sequences(tmp_path / "missing") == []


def test_load_sequence(seq_dir):
    _, d, heights, pitches, frames = seq_dir
    s = load_sequence(d)
    assert s.name == "Office_P0000" and len(s) == 4
    assert list(s.frame_ids) == frames
    assert list(s.depth_frame_ids) == [3, 5]
    assert s.intrinsics.width == W and np.isclose(s.intrinsics.fx, W / 2)
    assert np.isclose(s.floor_z_ned, FLOOR_Z, atol=0.02)
    assert s.floor_fit_spread_m < 0.02
    assert np.allclose(s.camera_height_m, heights[frames], atol=0.02)
    assert np.allclose(s.pitch_rad, pitches[frames], atol=1e-6)
    assert np.allclose(s.roll_rad, 0.0, atol=1e-6)
    f0 = s.frames[0]
    assert f0.dtype == np.uint8 and f0.shape == (H, W, 3)
    m = s.floor_masks[1]
    assert m.dtype == bool and m.shape == (H, W)
    assert m[-1].all() and not m[0].any()  # floor at the bottom, wall at the top
    assert len(s.floor_masks[1:3]) == 2
    cam = s.camera(0)
    assert isinstance(cam, CameraPose) and np.isclose(cam.height_m, heights[3], atol=0.02)
    occ = mask_to_bev(s.floor_masks[0], s.intrinsics, cam, px_per_m=10, extent_m=(4, 4), max_range_m=3.5)
    assert occ.dtype == bool and (~occ).any() and occ.any()


def test_load_sequence_errors(seq_dir):
    _, d, *_ = seq_dir
    (d / "seg_label_map.json").write_text(json.dumps({"name_map": {"wall": WALL}}))
    with pytest.raises(ValueError, match="no floor class"):
        load_sequence(d)


def test_missing_depth_is_an_error(seq_dir):
    _, d, *_ = seq_dir
    for p in (d / "depth").glob("*.png"):
        p.unlink()
    with pytest.raises(ValueError, match="no depth frame"):
        load_sequence(d)
