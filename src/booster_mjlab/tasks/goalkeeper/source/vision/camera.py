"""K1 head RGB-D camera model shared by rendering, geometric simulation and deployment.

Intrinsics follow the known RGB stream (544x488, HFOV 105 deg, square pixels).
The mount pose on Head_2 is a placeholder until the real extrinsics are measured.

Camera frame convention everywhere in this package is OpenCV: x right, y down,
z forward; pixel u = cx + fx * x / z, v = cy + fy * y / z.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

WIDTH, HEIGHT = 544, 488
HFOV_DEG = 105.0
FX = FY = WIDTH / 2 / math.tan(math.radians(HFOV_DEG / 2))  # ~208.7 px
CX, CY = WIDTH / 2, HEIGHT / 2
VFOV_DEG = math.degrees(2 * math.atan(HEIGHT / 2 / FY))  # ~98.9 deg
CAMERA_NAME = "head_cam"
PARENT_BODY = "Head_2"
# Front face of the Head_2 mesh, level and looking along the head +x axis.
MOUNT_POS = (0.065, 0.0, 0.08)
# Columns are the OpenCV camera axes expressed in the Head_2 frame:
# x_cv = -y_head (right), y_cv = -z_head (down), z_cv = +x_head (forward).
BODY_FROM_CV = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))


def mount_quat() -> tuple[float, float, float, float]:
    """MuJoCo camera (looks along -z, +y up) orientation in Head_2, wxyz."""
    body_from_mj = np.array(BODY_FROM_CV) @ np.diag([1.0, -1.0, -1.0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, body_from_mj.flatten())
    return tuple(float(q) for q in quat)


def head_camera_cfg(
    data_types=("rgb", "depth", "segmentation"), width=WIDTH, height=HEIGHT
):
    from mjlab.sensor.camera_sensor import CameraSensorCfg

    # Same focal geometry at any resolution.
    fy = FY * width / WIDTH
    return CameraSensorCfg(
        name=CAMERA_NAME,
        parent_body=f"robot/{PARENT_BODY}",
        pos=MOUNT_POS,
        quat=mount_quat(),
        fovy=math.degrees(2 * math.atan(height / 2 / fy)),
        width=width,
        height=height,
        data_types=tuple(data_types),
    )


def camera_pose(head_pos: torch.Tensor, head_rot: torch.Tensor):
    """World pose of the OpenCV camera frame from the Head_2 pose (rot: [N,3,3])."""
    mount = head_pos.new_tensor(MOUNT_POS)
    body_from_cv = head_pos.new_tensor(BODY_FROM_CV)
    return head_pos + head_rot @ mount, head_rot @ body_from_cv


def project(p_cam: torch.Tensor):
    """Pixel centre (u, v) of camera-frame points [..., 3]."""
    z = p_cam[..., 2].clamp_min(1e-6)
    return CX + FX * p_cam[..., 0] / z, CY + FY * p_cam[..., 1] / z


def unproject(u: torch.Tensor, v: torch.Tensor, z: torch.Tensor):
    return torch.stack(((u - CX) * z / FX, (v - CY) * z / FY, z), -1)


def mujoco_cam_to_cv(rot_mj: torch.Tensor) -> torch.Tensor:
    """World rotation of a MuJoCo camera ([..., 3, 3]) to the OpenCV frame."""
    return rot_mj * rot_mj.new_tensor([1.0, -1.0, -1.0])
