"""Frame clock + measurement front end + tracker, independent of the simulator.

    perception = BallPerception(n, device, cfg)
    perception.reset(ids, now)
    perception.step(now, dt, cam_pos, cam_rot, capture)   # every policy step
    perception.position, perception.valid                # tracking frame

`capture(mask)` returns (pos_cam [N,3], std [N,3], detected [N]) for the
current images or state; rows outside mask are ignored. The camera pose is the
one at capture time, so it is stored with the frame.
"""

from __future__ import annotations

import torch

from .config import VisionCfg
from .frames import FrameClock
from .measurement import PAYLOAD
from .tracker import BallTracker


class BallPerception:
    def __init__(self, num_envs: int, device, cfg: VisionCfg | None = None):
        self.cfg = cfg or VisionCfg()
        self.n, self.device = num_envs, device
        # Frame payload: camera-frame measurement + camera pose (pos 3, rot 9).
        self.clock = FrameClock(num_envs, PAYLOAD + 12, device, self.cfg.timing)
        self.tracker = BallTracker(num_envs, device, self.cfg.tracker)
        self.frames = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids, now):
        self.clock.reset(ids, now, self.cfg.randomize)
        self.tracker.reset(ids)
        self.frames[ids] = 0

    def step(self, now, dt, cam_pos, cam_rot, capture):
        self.tracker.predict(dt)
        due = self.clock.due(now)
        if due.any():
            pos, std, detected = capture(due)
            payload = torch.cat(
                (pos, std, detected[:, None].float(), cam_pos, cam_rot.flatten(1)), -1
            )
            self.clock.push(due, now, payload)
        payload, stamp, delivered = self.clock.pop(now)
        pos, std = payload[:, :3], payload[:, 3:6]
        use = delivered & (payload[:, 6] > 0.5)
        if use.any():
            c_pos, c_rot = payload[:, 7:10], payload[:, 10:19].reshape(-1, 3, 3)
            z = c_pos + (c_rot @ pos[:, :, None])[:, :, 0]
            R = (
                c_rot
                @ torch.diag_embed(std.square().clamp_min(1e-8))
                @ c_rot.transpose(1, 2)
            )
            self.tracker.update(z, R, now - stamp, use)
        self.frames += use.long()
        return use

    @property
    def position(self):
        return self.tracker.position

    @property
    def valid(self):
        return self.tracker.valid
