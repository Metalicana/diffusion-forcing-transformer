"""CaM UE coordinates -> DFoT normalized intrinsics plus world-to-camera."""
import json
from pathlib import Path
import numpy as np


def nominal_intrinsics(width, height, horizontal_fov_deg=52.67):
    """Full-view resize to 256 square, in DFoT's image-edge coordinates.

    Pixel centers are i+0.5; centered principal point is (W/2,H/2).
    PIL full-image resize scales this coordinate system without an offset.
    """
    if width <= 0 or height <= 0 or not 0 < horizontal_fov_deg < 180:
        raise ValueError('Invalid image dimensions or horizontal FOV')
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    native = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    transform = np.diag([256 / width, 256 / height, 1])
    resized = transform @ native
    return dict(native_size=[width, height], target_size=[256, 256],
                K_native=native.tolist(), resize_transform=transform.tolist(),
                K_input=resized.tolist(),
                normalized=[resized[0, 0] / 256, resized[1, 1] / 256, .5, .5],
                coordinate_convention='image edges at 0/W and 0/H; pixel centers i+0.5')


def convert(path, start, count, intrinsics, scale=100.0):
    if len(intrinsics) != 4 or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0:
        raise ValueError('Supply verified normalized source-image fx,fy,cx,cy')
    data = json.loads(Path(path).read_text())['CineCameraActor']
    keys = sorted(data, key=int)[start:start + count]
    if len(keys) != count:
        raise ValueError('Trajectory is shorter than requested output')
    # Columns map OpenCV camera (right, down, forward) into UE (forward, right, up).
    basis = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    poses = []
    for key in keys:
        frame = data[key]
        p, r, y = np.deg2rad(frame['rotation'])
        cp, sp, cr, sr, cy, sy = np.cos(p), np.sin(p), np.cos(r), np.sin(r), np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        c2w = np.eye(4)
        # Change both world and camera basis to retain a proper rotation.
        c2w[:3, :3] = basis.T @ (rz @ ry @ rx) @ basis
        c2w[:3, 3] = basis.T @ np.asarray(frame['position']) / scale
        poses.append(np.linalg.inv(c2w)[:3].reshape(-1))
    vectors = np.concatenate([np.tile(intrinsics, (count, 1)), poses], axis=1)
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite camera conditions')
    return vectors.astype(np.float32), keys
