"""Concrete data and environment interfaces for categorical goalkeeper AMP."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from mjlab.managers import SceneEntityCfg

from booster_mjlab.motion.motion_data import MotionFile
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER


MOTION_NAMES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
MOTION_ID_MAP = {name: index for index, name in enumerate(MOTION_NAMES)}


class GoalkeeperExpertSampler:
    """Sample chronological two-frame joint windows from six motion clips."""

    joint_names = K1_JOINT_ORDER

    def __init__(self, dataset_root: str | Path, device: torch.device | str):
        self.dataset_root = Path(dataset_root)
        self.device = torch.device(device)
        self._validate_manifest()
        self._positions: list[torch.Tensor] = []
        self._fps: list[float] = []

        for motion_name in MOTION_NAMES:
            path = self.dataset_root / f"{motion_name}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing goalkeeper expert clip: {path}")
            motion = MotionFile.load(path)
            positions = torch.as_tensor(
                motion.dof_pos, dtype=torch.float32, device=self.device
            )
            if positions.ndim != 2 or positions.shape[1] != len(K1_JOINT_ORDER):
                raise ValueError(
                    f"{path} must contain (T, {len(K1_JOINT_ORDER)}) dof_pos"
                )
            if positions.shape[0] < 2:
                raise ValueError(f"{path} must contain at least two frames")
            if not torch.isfinite(positions).all().item():
                raise ValueError(f"{path} contains non-finite joint positions")
            if not math.isfinite(motion.fps) or motion.fps <= 0.0:
                raise ValueError(f"{path} has an invalid fps: {motion.fps}")
            self._positions.append(positions)
            self._fps.append(float(motion.fps))

    def _validate_manifest(self) -> None:
        manifest_path = self.dataset_root.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing goalkeeper motion manifest: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid goalkeeper manifest: {manifest_path}") from error

        if tuple(manifest.get("joint_names", ())) != tuple(K1_JOINT_ORDER):
            raise ValueError("Manifest joint_names must match K1_JOINT_ORDER")
        if manifest.get("motion_id_map") != MOTION_ID_MAP:
            raise ValueError("Manifest motion_id_map does not match the AMP contract")

        reports = manifest.get("reports")
        if not isinstance(reports, list) or len(reports) != len(MOTION_NAMES):
            raise ValueError("Manifest reports must contain all six clip reports")
        reports_by_clip = {
            report.get("clip"): report for report in reports if isinstance(report, dict)
        }
        if set(reports_by_clip) != set(MOTION_NAMES):
            raise ValueError(
                "Manifest reports must contain exactly the six motion clips"
            )
        rejected = [
            name
            for name in MOTION_NAMES
            if reports_by_clip[name].get("acceptance", {}).get("status")
            != "kinematic_checks_passed"
        ]
        if rejected:
            raise ValueError(
                "Goalkeeper clips did not pass kinematic checks: " + ", ".join(rejected)
            )

    def sample(self, motion_ids: torch.Tensor, *, frame_dt: float) -> torch.Tensor:
        if (
            motion_ids.ndim != 1
            or motion_ids.dtype != torch.int64
            or motion_ids.device != self.device
        ):
            raise ValueError("motion_ids must be int64 (B,) on the sampler device")
        if ((motion_ids < 0) | (motion_ids >= len(MOTION_NAMES))).any().item():
            raise ValueError("motion_ids must be in [0, 5]")
        if not math.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError("frame_dt must be finite and positive")

        windows = torch.empty(
            (motion_ids.shape[0], 2, len(K1_JOINT_ORDER)),
            dtype=torch.float32,
            device=self.device,
        )
        for motion_id, (positions, fps) in enumerate(
            zip(self._positions, self._fps, strict=True)
        ):
            rows = (motion_ids == motion_id).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            duration = (positions.shape[0] - 1) / fps
            if frame_dt > duration:
                raise ValueError(
                    f"{MOTION_NAMES[motion_id]} is shorter than frame_dt={frame_dt}"
                )
            start_time = torch.rand(rows.numel(), device=self.device) * (
                duration - frame_dt
            )
            times = torch.stack((start_time, start_time + frame_dt), dim=1)
            frame = times * fps
            lower = torch.floor(frame).to(torch.long)
            upper = torch.clamp(lower + 1, max=positions.shape[0] - 1)
            alpha = (frame - lower.to(frame.dtype)).unsqueeze(-1)
            sampled = positions[lower] * (1.0 - alpha) + positions[upper] * alpha
            windows[rows] = sampled
        return windows


def create_goalkeeper_expert_sampler(
    *, dataset_root: str | Path, device: torch.device | str
) -> GoalkeeperExpertSampler:
    return GoalkeeperExpertSampler(dataset_root=dataset_root, device=device)


def amp_joint_positions(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=K1_JOINT_ORDER, preserve_order=True
    ),
) -> torch.Tensor:
    """Return raw absolute K1 joint angles in the AMP contract order."""
    robot = env.scene[asset_cfg.name]
    result = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if result.shape != (env.num_envs, len(K1_JOINT_ORDER)):
        raise RuntimeError(
            f"Expected AMP joint positions {(env.num_envs, len(K1_JOINT_ORDER))}, "
            f"got {tuple(result.shape)}"
        )
    return result


def get_amp_motion_ids(env) -> torch.Tensor:
    """Read the per-episode shot category without exposing private storage."""
    shots = getattr(env, "_full_goal_shots", None)
    if not isinstance(shots, dict) or "region" not in shots:
        raise RuntimeError("Goalkeeper shots have not been initialized")
    motion_ids = shots["region"]
    if (
        motion_ids.shape != (env.num_envs,)
        or motion_ids.dtype != torch.int64
        or motion_ids.device != torch.device(env.device)
        or ((motion_ids < 0) | (motion_ids >= len(MOTION_NAMES))).any().item()
    ):
        raise RuntimeError("Invalid goalkeeper motion ids")
    return motion_ids
