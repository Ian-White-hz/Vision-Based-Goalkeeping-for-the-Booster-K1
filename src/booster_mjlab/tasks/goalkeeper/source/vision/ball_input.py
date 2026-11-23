"""The only part of the vision package that reads the environment.

`VisionBallInput()` returns the actor's ball input, (ball_local [N,3], seen [N]):
the tracked ball relative to the torso in the torso frame, zero where the
tracker has no valid estimate. Modes:

- "synthetic": geometric measurement model (training, no rendering).
- "rendered": head camera images -> detector -> localize (evaluation). Needs
  the camera sensor in the scene (env_cfg(ball_input="rendered")).

The tracking frame is the world frame shifted by a random-walk odometry error,
since the robot only knows its own position from leg odometry.
"""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

from . import camera
from .config import VisionCfg
from .detectors import SegmentationDetector
from .measurement import GeometricMeasurement, localize
from .occlusion import ARM_CHAINS, occluded_fraction, occluder_points
from .pipeline import BallPerception

MODES = ("synthetic", "rendered")


class VisionBallInput:
    def __init__(self, env, mode: str, cfg: VisionCfg | None = None, detector=None):
        if mode not in MODES:
            raise ValueError(
                f"ball_input must be 'truth' or one of {MODES}, got {mode!r}"
            )
        self.env, self.mode = env, mode
        self.cfg = cfg or VisionCfg()
        n, dev = env.num_envs, env.device
        self.robot, self.ball = env.scene["robot"], env.scene["ball"]
        self.head = self.robot.find_bodies(camera.PARENT_BODY)[0][0]
        self.chains = [
            self.robot.find_bodies(c, preserve_order=True)[0] for c in ARM_CHAINS
        ]
        self.perception = BallPerception(n, dev, self.cfg)
        self.drift = torch.zeros(n, 3, device=dev)
        self.measurement = GeometricMeasurement(n, dev, self.cfg.measurement)
        if mode == "rendered":
            self.sensor = env.scene.sensors[camera.CAMERA_NAME]
            if detector is None:
                geom = env.sim.mj_model.geom("ball/football_geom").id
                detector = SegmentationDetector(geom, int(mujoco.mjtObj.mjOBJ_GEOM))
        self.detector = detector
        self.ball_local = torch.zeros(n, 3, device=dev)
        self.seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.true_local = torch.zeros(n, 3, device=dev)
        self.last_capture = None

    def camera_pose(self):
        """True world pose of the OpenCV camera frame from the Head_2 pose."""
        d = self.robot.data
        rot = matrix_from_quat(d.body_link_quat_w[:, self.head])
        return camera.camera_pose(d.body_link_pos_w[:, self.head], rot)

    def _capture(self, cam_pos, cam_rot):
        ball = self.ball.data.root_link_pos_w

        p_cam = (cam_rot.transpose(1, 2) @ (ball - cam_pos)[:, :, None])[:, :, 0]

        def synthetic(mask):
            spheres = occluder_points(self.robot.data.body_link_pos_w, self.chains)
            occ = occluded_fraction(
                cam_pos,
                ball,
                spheres,
                self.cfg.occluder_radius,
                self.cfg.measurement.ball_radius,
            )
            return self.measurement(p_cam, occ)

        def rendered(mask):
            data = self.sensor.data
            box, detected = self.detector(data.rgb, data.segmentation)
            return localize(
                box, detected & mask, data.depth[..., 0], self.cfg.measurement
            )

        front = synthetic if self.mode == "synthetic" else rendered

        def capture(mask):
            pos, std, detected = front(mask)
            # Raw frame vs truth, for calibration and evaluation.
            self.last_capture = (mask, pos, std, detected, p_cam)
            return pos, std, detected

        return capture

    def __call__(self):
        env, dt = self.env, self.env.step_dt
        now = env.episode_length_buf.float() * dt
        fresh = torch.nonzero(env.episode_length_buf == 0).flatten()
        if len(fresh):
            self.perception.reset(fresh, now)
            self.measurement.reset(fresh, self.cfg.randomize)
            self.drift[fresh] = 0.0
        if self.cfg.randomize:
            self.drift += (
                torch.randn_like(self.drift) * self.cfg.odom_drift * math.sqrt(dt)
            )
        cam_pos, cam_rot = self.camera_pose()
        self.last_capture = None
        self.perception.step(
            now, dt, cam_pos + self.drift, cam_rot, self._capture(cam_pos, cam_rot)
        )

        d = self.robot.data
        torso = d.root_link_pos_w + self.drift
        self.seen = self.perception.valid.clone()
        local = quat_apply_inverse(d.root_link_quat_w, self.perception.position - torso)
        self.ball_local = local * self.seen[:, None]
        self.true_local = quat_apply_inverse(
            d.root_link_quat_w, self.ball.data.root_link_pos_w - d.root_link_pos_w
        )
        return self.ball_local, self.seen


def make_ball_input(env, mode: str, randomize: bool = True, **kwargs):
    cfg = kwargs.pop("cfg", None) or VisionCfg(randomize=randomize)
    return VisionBallInput(env, mode, cfg=cfg, **kwargs)
