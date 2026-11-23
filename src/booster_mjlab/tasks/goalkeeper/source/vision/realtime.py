"""On-robot ball input with the same detector, localization and tracker as simulation.

No ROS or driver code: the caller feeds frames and asks for the actor input.

    ball = RealtimeBallInput(YoloDetector("best.pt"))
    ball.reset()                                   # at the start of each shot
    ball.on_frame(rgb, depth, stamp, cam_pos, cam_rot)   # camera callback
    local, seen = ball.query(now, torso_pos, torso_rot)  # every policy step (50 Hz)
    obs[:3] = local if seen else 0

Frames: rgb uint8 [H,W,3] (RGB order) and depth float [H,W] in metres,
planar and registered to the RGB image. cam_pos / cam_rot are the OpenCV
camera pose in the odometry frame (gravity-aligned, z up) at `stamp`, e.g.
torso pose from IMU + leg odometry composed with head FK and the camera mount
(camera.camera_pose). torso_* are the torso pose in the same frame at `now`.
Times are seconds on one clock.
"""

from __future__ import annotations

import torch

from .config import VisionCfg
from .measurement import localize
from .tracker import BallTracker


class RealtimeBallInput:
    def __init__(self, detector, cfg: VisionCfg | None = None, device="cpu"):
        self.cfg = cfg or VisionCfg()
        self.detector, self.device = detector, device
        self.tracker = BallTracker(1, device, self.cfg.tracker)
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.time = None
        self.pending = []

    def on_frame(self, rgb, depth, stamp: float, cam_pos, cam_rot):
        rgb = torch.as_tensor(rgb, device=self.device)[None]
        depth_t = (
            None
            if depth is None
            else torch.as_tensor(depth, device=self.device, dtype=torch.float32)[None]
        )
        box, detected = self.detector(rgb, None)
        pos, std, ok = localize(
            box.to(self.device), detected.to(self.device), depth_t, self.cfg.measurement
        )
        if not bool(ok[0]):
            return False
        c_pos = torch.as_tensor(cam_pos, device=self.device, dtype=torch.float32)
        c_rot = torch.as_tensor(cam_rot, device=self.device, dtype=torch.float32)
        z = c_pos + c_rot @ pos[0]
        R = c_rot @ torch.diag(std[0].square().clamp_min(1e-8)) @ c_rot.T
        self.pending.append((float(stamp), z, R))
        return True

    def query(self, now: float, torso_pos, torso_rot):
        if self.time is None:
            self.time = now
        self.tracker.predict(max(now - self.time, 0.0))
        self.time = now
        for stamp, z, R in sorted(self.pending, key=lambda p: p[0]):
            lag = torch.tensor([max(now - stamp, 0.0)], device=self.device)
            self.tracker.update(
                z[None],
                R[None],
                lag,
                torch.ones(1, dtype=torch.bool, device=self.device),
            )
        self.pending.clear()
        seen = bool(self.tracker.valid[0])
        t_pos = torch.as_tensor(torso_pos, device=self.device, dtype=torch.float32)
        t_rot = torch.as_tensor(torso_rot, device=self.device, dtype=torch.float32)
        local = t_rot.T @ (self.tracker.position[0] - t_pos)
        return (local if seen else torch.zeros(3, device=self.device)), seen
