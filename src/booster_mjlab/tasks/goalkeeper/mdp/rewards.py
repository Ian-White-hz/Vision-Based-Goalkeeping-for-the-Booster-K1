from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab.utils.lab_api.math import quat_apply_inverse

from .outcomes import update_goalkeeper_outcome_state
from .interception import lateral_velocity_tracking_reward, interception_features
from .shot_state import (
    get_recovery_mask,
    stopball_transition as stopball_transition,  # Public compatibility export.
    get_shot_outcome_state,
)

import math
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_conceded_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One goal event per shot, normalized so its weighted cost is dt invariant.

    Repeated reads in the same control step return the same pulse; this is
    intentionally not consumed by a metric or another reward reader.
    """
    state = update_goalkeeper_outcome_state(env)
    event = state.goal_conceded & (state.goal_conceded_step == env.common_step_counter)
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    return event.float() / scale


def fall_termination_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fixed cost on the excessive-tilt terminal step, independent of dt.

    Reuse that termination's configured predicate, never aggregate done flags.
    Reading the current pose also avoids cached terminal masks after auto-reset.
    The environment requires reset before another step after termination, so
    this event is charged once per failed episode; repeated reads are stable.
    """
    term = env.termination_manager.get_term_cfg("excessive_body_tilt")
    fallen = term.func(env, **term.params)
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    return fallen.float() / scale


def body_tilt_cost(
    projected_gravity_b: torch.Tensor,
    *,
    deadband_deg: float = 20.0,
    full_penalty_deg: float = 60.0,
) -> torch.Tensor:
    """Bounded squared trunk tilt beyond a deadband; unit gravity input (B, 3)."""
    if projected_gravity_b.ndim != 2 or projected_gravity_b.shape[-1] != 3:
        raise ValueError("Expected projected gravity shape (B, 3)")
    if not (
        math.isfinite(deadband_deg)
        and math.isfinite(full_penalty_deg)
        and 0.0 <= deadband_deg < full_penalty_deg <= 180.0
    ):
        raise ValueError("Require 0 <= deadband < full penalty <= 180 degrees")

    tilt = torch.acos((-projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    lower, upper = math.radians(deadband_deg), math.radians(full_penalty_deg)
    return ((tilt - lower) / (upper - lower)).clamp(0.0, 1.0).square()


def body_tilt_from_env(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    deadband_deg: float = 20.0,
    full_penalty_deg: float = 60.0,
) -> torch.Tensor:
    """Continuous whole-episode tilt cost; RewardManager applies dt scaling."""
    return body_tilt_cost(
        env.scene[robot_name].data.projected_gravity_b,
        deadband_deg=deadband_deg,
        full_penalty_deg=full_penalty_deg,
    )


def heading_deviation_cost(
    quaternion_w: torch.Tensor,
    reference_quaternion_w: torch.Tensor,
    *,
    deadband_deg: float = 20.0,
    full_penalty_deg: float = 90.0,
) -> torch.Tensor:
    """Bounded squared heading excess, with a shortest-angle deadband."""
    if (
        quaternion_w.ndim != 2 or quaternion_w.shape[-1] != 4
        or reference_quaternion_w.shape != quaternion_w.shape
    ):
        raise ValueError("Quaternions must have matching (num_envs, 4) shapes")
    if (
        not math.isfinite(deadband_deg) or not math.isfinite(full_penalty_deg)
        or not 0.0 <= deadband_deg < full_penalty_deg <= 180.0
    ):
        raise ValueError("Require 0 <= heading deadband < full penalty <= 180")

    def heading(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    delta = heading(quaternion_w) - heading(reference_quaternion_w)
    error = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
    lower, upper = math.radians(deadband_deg), math.radians(full_penalty_deg)
    return ((error - lower) / (upper - lower)).clamp(0.0, 1.0).square()


def excessive_yaw_rate_cost(
    angular_velocity_z_w: torch.Tensor,
    *,
    deadband: float = 0.5,
    full_penalty_rate: float = 3.0,
) -> torch.Tensor:
    """Bounded squared world-vertical angular-speed excess, in rad/s."""
    if angular_velocity_z_w.ndim != 1:
        raise ValueError("Vertical angular velocity must have shape (num_envs,)")
    if (
        not math.isfinite(deadband) or not math.isfinite(full_penalty_rate)
        or not 0.0 <= deadband < full_penalty_rate
    ):
        raise ValueError("Require 0 <= yaw-rate deadband < full penalty rate")
    excess = (angular_velocity_z_w.abs() - deadband) / (full_penalty_rate - deadband)
    return excess.clamp(0.0, 1.0).square()


def heading_deviation_from_env(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    deadband_deg: float = 20.0,
    full_penalty_deg: float = 90.0,
) -> torch.Tensor:
    """Keep the reset configuration's guard heading; no new Actor observation."""
    robot = env.scene[robot_name]
    return heading_deviation_cost(
        robot.data.root_link_quat_w,
        robot.data.default_root_state[:, 3:7],
        deadband_deg=deadband_deg,
        full_penalty_deg=full_penalty_deg,
    )


def excessive_yaw_rate_from_env(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    deadband: float = 0.5,
    full_penalty_rate: float = 3.0,
) -> torch.Tensor:
    robot = env.scene[robot_name]
    return excessive_yaw_rate_cost(
        robot.data.root_link_ang_vel_w[:, 2],
        deadband=deadband,
        full_penalty_rate=full_penalty_rate,
    )


class MotorLimitPenalty:

    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        robot = env.scene[cfg.params["robot_name"]]
        kind = cfg.params["kind"]

        if kind not in ("velocity", "torque"):
            raise ValueError(f"Unknown motor limit kind: {kind}")

        joint_ids = []
        limits = []

        for actuator in robot.actuators:
            if actuator.cfg.transmission_type != TransmissionType.JOINT:
                raise ValueError("Only joint actuators are supported for motor limit penalties.")

            ids = actuator.target_ids.tolist()

            if kind == "velocity":
                limit = actuator.cfg.motor.velocity_limit
            else:
                limit = actuator.cfg.effort_limit

            if limit is None or not math.isfinite(limit) or limit <= 0.0:
                raise ValueError(f"{kind} limit must be finite and positive")

            joint_ids.extend(ids)
            limits.extend([limit] * len(ids))

        if not joint_ids:
            raise ValueError("No actuated joints found")

        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("Multiple actuators target the same joint")

        reference = robot.data.joint_vel

        self._joint_ids = torch.tensor(
            joint_ids,
            dtype=torch.long,
            device=reference.device
        )
        self._limits = torch.tensor(
            limits,
            dtype=torch.float32,
            device=reference.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_name: str,
        kind: str,
        soft_ratio:float,
    ) -> torch.Tensor:
        robot = env.scene[robot_name]

        if kind == "velocity":
            value = robot.data.joint_vel[:, self._joint_ids]
        else:
            value = robot.data.qfrc_actuator[:, self._joint_ids]

        excess = value.abs() - soft_ratio * self._limits

        if kind == "velocity":
            excess = torch.clamp(excess, min=0.0, max=1.0)
        else:
            excess = torch.clamp(excess, min=0.0)

        return torch.sum(excess, dim=1)

class ActuatedJointAccelerationPenalty:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        robot = env.scene[cfg.params["robot_name"]]
        self._joint_ids = _actuated_joint_ids(robot)

        velocity = robot.data.joint_vel[:, self._joint_ids]

        if env.step_dt <= 0:
            raise ValueError("Step dt must be positive")

        self._previous_veloctiy = torch.zeros_like(velocity)
        self._cached_cost = torch.zeros_like(velocity[:, 0])
        self._last_step = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_name: str,
    ) -> torch.Tensor:
        if self._last_step != env.common_step_counter:
            robot = env.scene[robot_name]
            velocity = robot.data.joint_vel[:, self._joint_ids]

            acceleration = (
                velocity - self._previous_veloctiy
            ) / env.step_dt

            self._cached_cost = torch.sum(torch.square(acceleration), dim=1)
            self._previous_veloctiy = velocity
            self._last_step = env.common_step_counter
        
        return self._cached_cost

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._previous_veloctiy[env_ids] = 0.0
        self._cached_cost[env_ids] = 0.0

class NormalizedJointTorquePenalty:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        robot = env.scene[cfg.params["robot_name"]]
        self._joint_ids = _actuated_joint_ids(robot)

        gains = []

        for actuator in robot.actuators:
            kp = actuator.cfg.stiffness

            if not math.isfinite(kp) or kp <= 0.0:
                raise ValueError("Stiffness must be finite and positive")

            gains.extend([kp] * actuator.target_ids.numel())

        reference = robot.data.joint_vel
        self._kp = torch.tensor(
            gains,
            dtype=reference.dtype,
            device=reference.device,
        )

    def __call__(
        self, 
        env: ManagerBasedRlEnv,
        robot_name: str,
    ) -> torch.Tensor:
        robot = env.scene[robot_name]
        torque = robot.data.qfrc_actuator[:, self._joint_ids]

        return torch.sum(torch.square(torque / self._kp), dim=1)
        
class FeetSlippageReward:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        robot = env.scene[cfg.params["robot_name"]]
        sensor = env.scene[cfg.params["sensor_name"]]

        foot_names = sensor.primary_names
        foot_ids, resolved_names = robot.find_bodies(
            foot_names,
            preserve_order=True,
        )

        if len(foot_ids) != 2:
            raise ValueError("Expected exactly two foot bodies")

        if list(resolved_names) != list(foot_names):
            raise ValueError("Foot names do not match resolved names")

        self._foot_ids = foot_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_name: str,
        sensor_name: str,
        contact_threshold: float = 1.0,
        coefficient: float = 10.0,
    ) -> torch.Tensor:
        robot = env.scene[robot_name]
        sensor = env.scene[sensor_name]

        foot_velocity_w = robot.data.body_link_lin_vel_w[:, self._foot_ids, :]
        foot_force_w = sensor.data.force

        if foot_force_w is None:
            raise ValueError("Foot contact forces are not available")

        return feet_slippage_exp(
            foot_velocity_w=foot_velocity_w,
            foot_force_w=foot_force_w,
            contact_threshold=contact_threshold,
            coefficient=coefficient,
        )

class JumpLandingReward:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self._was_supported = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._has_in_air = torch.zeros_like(self._was_supported)
        self._cached_reward = torch.zeros(
            env.num_envs,
            device=env.device,
        )
        self._last_update_step = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        for value in (
            self._was_supported,
            self._has_in_air,
            self._cached_reward,
        ):
            value[env_ids] = 0

        self._last_update_step[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_name: str,
        sensor_name: str,
        minimum_rise: float = 0.08,
        force_sign: float = -1.0,
        contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        robot = env.scene[robot_name]
        sensor = env.scene[sensor_name]

        force = sensor.data.force
        found = sensor.data.found
        if force is None or found is None:
            raise RuntimeError("Landing reward requires found and force")

        # 当前脚—地面传感器的支撑 Fz 为负，在适配层转换。
        support_force = force_sign * force

        region = env._full_goal_shots["region"]
        jump_mask = (region == 2) | (region == 3)

        both_supported = (
            (support_force[..., 2] > contact_threshold)
            & (found > 0)
        ).all(dim=1)

        no_foot_contact = ~(found > 0).any(dim=1)

        height = (
            robot.data.root_link_pos_w[:, 2]
            - env.scene.env_origins[:, 2]
        )
        default_height = robot.data.default_root_state[:, 2]

        takeoff = (
            jump_mask
            & self._was_supported
            & no_foot_contact
            & (height - default_height >= minimum_rise)
        )

        fresh = (
            self._last_update_step != env.common_step_counter
        )

        self._was_supported |= fresh & both_supported
        self._has_in_air |= fresh & takeoff

        reward = jump_landing_reward(
            foot_force_w=support_force,
            has_in_air=self._has_in_air,
            jump_mask=jump_mask,
            contact_threshold=contact_threshold,
        )

        self._cached_reward.copy_(
            torch.where(fresh, reward, self._cached_reward)
        )
        self._last_update_step.fill_(env.common_step_counter)

        return self._cached_reward

def base_ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    '''惩罚机器人绕 x 和 y 轴的角速度, shape (B,)'''
    robot = env.scene[robot_name]
    angular_velocity_b = robot.data.root_link_ang_vel_b

    return torch.sum(
        torch.square(angular_velocity_b[:, :2]),
        dim=1,
    )

def base_backward_velocity_penalty(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    '''惩罚机器人向后运动, shape (B,)'''
    robot = env.scene[robot_name]
    velocity_x_b = robot.data.root_link_lin_vel_b[:, 0]

    return torch.clamp(
        -velocity_x_b,
        min=0.0,
        max=1.0,
    )

def base_line_deviation(
    env:ManagerBasedRlEnv,
    robot_name:str = "robot",
    reference_x: float = 0.0,
    deadband: float = 0.2,
    max_distance: float = 1.2,
):
    """Deviation from a reference plane along world x, shape (B,)"""
    robot = env.scene[robot_name]

    position_x = (
        robot.data.root_link_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )
    distance = torch.abs(position_x - reference_x)

    return torch.clamp(
        distance,
        min=deadband,
        max=max_distance,
    ) - deadband

def knee_height_below_minimum(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.15,
):
    """Whether any selected knee is too low above the flat ground, shape (B,)."""
    robot = env.scene[asset_cfg.name]

    knee_height = (
        robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
        - env.scene.env_origins[:, 2:3]
    )

    return (
        torch.min(knee_height, dim=1).values < minimum_height
    ).float()

def feet_orientation_exp(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    coefficient: float = 5.0
) -> torch.Tensor:
    '''奖励机器人保持脚部朝向正确, shape (B,)'''
    robot = env.scene[asset_cfg.name]

    foot_quanternion = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]

    gravity_w = robot.data.gravity_vec_w[:, None, :].expand(
        -1, foot_quanternion.shape[1], -1
    )

    gravity_foot = quat_apply_inverse(
        foot_quanternion,
        gravity_w,
    )

    error = torch.sum(torch.square(gravity_foot[..., :2]), dim=(1, 2))

    return torch.exp(-coefficient * error)

def post_orientation_exp(
    env: ManagerBasedRlEnv,
    recovery_mask: torch.Tensor,
    robot_name: str = "robot",
    coefficient: float = 3.0,
)-> torch.Tensor:
    '''鼓励机器人恢复到正确姿态, shape (B,)'''
    robot = env.scene[robot_name]
    gravity_b = robot.data.projected_gravity_b

    error = torch.sum(torch.square(gravity_b[:, :2]), dim=1)

    return _masked_recovery_exp(
        error=error,
        recovery_mask=recovery_mask,
        coefficient=coefficient,
    )

def post_ang_vel_exp(
    env: ManagerBasedRlEnv,
    recovery_mask: torch.Tensor,
    robot_name: str = "robot",
    coefficient: float = 3.0,
) -> torch.Tensor:
    '''鼓励机器人恢复到正确角速度, shape (B,)'''
    robot = env.scene[robot_name]
    angular_velocity_b = robot.data.root_link_ang_vel_b

    error = torch.sum(torch.square(angular_velocity_b[:, :2]), dim=1)

    return _masked_recovery_exp(
        error=error,
        recovery_mask=recovery_mask,
        coefficient=coefficient,
    )

def post_lin_vel_exp(
    env: ManagerBasedRlEnv,
    recovery_mask: torch.Tensor,
    robot_name: str = "robot",
    coefficient: float = 3.0,
) -> torch.Tensor:
    '''鼓励机器人恢复到正确线速度, shape (B,)'''
    robot = env.scene[robot_name]
    velocity_x_b = robot.data.root_link_lin_vel_b[:, 0]

    error = torch.square(velocity_x_b)

    return _masked_recovery_exp(
        error=error,
        recovery_mask=recovery_mask,
        coefficient=coefficient,
    )

def post_upper_joint_position_exp(
    joint_position: torch.Tensor,
    default_joint_position: torch.Tensor,
    recovery_mask: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    '''鼓励机器人恢复到正确上半身关节位置, shape (B,)'''
    if (
        joint_position.ndim != 2
        or joint_position.shape != default_joint_position.shape
    ):
        raise ValueError("Joint positions must have matching shape (B, N)")

    error = (
        joint_position - default_joint_position
    ).square().sum(dim=1)

    return _masked_recovery_exp(
        error=error,
        recovery_mask=recovery_mask,
        coefficient=coefficient,
    )

def post_recovery_from_env(
    env: ManagerBasedRlEnv,
    kind: str,
    robot_name: str = "robot",
    ball_name: str = "ball",
    reference_x: float = 0.0,
    velocity_change_threshold: float = 2.0,
    coefficient: float = 3.0,
)  -> torch.Tensor:
    '''获取机器人恢复奖励, shape (B,)'''
    if kind == "orientation":
        reward_func = post_orientation_exp
    elif kind == "angular_velocity":
        reward_func = post_ang_vel_exp
    elif kind == "linear_velocity":
        reward_func = post_lin_vel_exp
    else:
        raise ValueError(f"Unknown recovery kind: {kind}")

    recovery_mask = get_recovery_mask(
        env=env,
        ball_name=ball_name,
        reference_x=reference_x,
        velocity_change_threshold=velocity_change_threshold,
    )

    return reward_func(
        env,
        recovery_mask=recovery_mask,
        robot_name=robot_name,
        coefficient=coefficient,
    )

def post_upper_joint_position_from_env(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    ball_name: str = "ball",
    reference_x: float = 0.0,
    velocity_change_threshold: float = 2.0,
    coefficient: float = 1.0,
) -> torch.Tensor:
    '''获取机器人上半身关节位置恢复奖励, shape (B,)'''
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    joint_position = robot.data.joint_pos[:, joint_ids]
    default_position = robot.data.default_joint_pos[:, joint_ids]

    recovery_mask = get_recovery_mask(
        env=env,
        ball_name=ball_name,
        reference_x=reference_x,
        velocity_change_threshold=velocity_change_threshold,
    )

    return post_upper_joint_position_exp(
        joint_position=joint_position,
        default_joint_position=default_position,
        recovery_mask=recovery_mask,
        coefficient=coefficient,
    )  

def feet_slippage_exp(
    foot_velocity_w: torch.Tensor,
    foot_force_w: torch.Tensor,
    contact_threshold: float = 1.0,
    coefficient: float = 10.0,
) -> torch.Tensor:
    '''惩罚脚部接触时的滑动, shape (B,)'''
    _check_foot_vectors(foot_velocity_w)
    _check_foot_vectors(foot_force_w)

    if foot_velocity_w.shape != foot_force_w.shape:
        raise ValueError("Foot velocity and force must have the same shape")

    speed = torch.linalg.vector_norm(foot_velocity_w, dim=-1)
    force = torch.linalg.vector_norm(foot_force_w, dim=-1)

    in_contact = force > contact_threshold

    contact_speed = torch.sum(
        speed * in_contact.to(dtype=speed.dtype), dim=1
    )

    return torch.exp(-coefficient * contact_speed)

def excessive_foot_contact(
    foot_force_w: torch.Tensor,
    max_contact_force: float = 1000.0,
) -> torch.Tensor:
    '''惩罚脚部接触力过大, shape (B,)'''
    _check_foot_vectors(foot_force_w)

    force = torch.linalg.vector_norm(foot_force_w, dim=-1)

    mean_force = torch.mean(force, dim=1)

    return (mean_force > max_contact_force).float()

def excessive_foot_contact_from_env(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_contact_force: float = 1000.0,
) -> torch.Tensor:
    '''惩罚脚部接触力过大, shape (B,)'''
    sensor = env.scene[sensor_name]
    foot_force_w = sensor.data.force

    if foot_force_w is None:
        raise ValueError("Foot contact forces are not available")

    return excessive_foot_contact(foot_force_w, max_contact_force)

def jump_landing_reward(
    foot_force_w: torch.Tensor,
    has_in_air: torch.Tensor,
    jump_mask: torch.Tensor,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    '''奖励机器人在跳跃后双脚着地, shape (B,)'''
    _check_foot_vectors(foot_force_w)
    batch_size = foot_force_w.shape[0]

    for flag in (has_in_air, jump_mask):
        if flag.dtype != torch.bool or flag.shape != (batch_size,):
            raise ValueError("has_in_air and jump_mask must be boolean tensors of shape (B,)")

    force_z = foot_force_w[..., 2]
    left = force_z[:, 0]
    right = force_z[:, 1]

    both_contact = (
        (left > contact_threshold) 
        & (right > contact_threshold)
        & has_in_air
    )

    one_contact = (
        ((left > contact_threshold) & (right < contact_threshold))
        | ((left < contact_threshold) & (right > contact_threshold))
    ) & has_in_air

    reward = (
        has_in_air.float()
        + 5.0 * both_contact.float()
        - one_contact.float()
    )

    return reward * jump_mask.to(dtype=reward.dtype)

def intercept_proximity_reward(
    distance: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    distance_sigma: float = 5.0,
    reach_threshold: float = 0.2,
) -> torch.Tensor:
    """有效拦截阶段内的距离奖励，返回 (B,)。"""
    if distance.ndim != 1:
        raise ValueError("distance must have shape (B,)")

    if (
        active_mask.shape != distance.shape
        or active_mask.dtype != torch.bool
        or active_mask.device != distance.device
    ):
        raise ValueError(
            "active_mask must be bool (B,) on the distance device"
        )

    if not math.isfinite(distance_sigma) or distance_sigma <= 0.0:
        raise ValueError("distance_sigma must be finite and positive")

    if not math.isfinite(reach_threshold) or reach_threshold < 0.0:
        raise ValueError("reach_threshold must be finite and nonnegative")

    proximity = torch.sigmoid(
        distance_sigma * (reach_threshold - distance)
    )

    return torch.where(
        active_mask,
        proximity,
        torch.zeros_like(proximity)
    )

def intercept_movement_reward(
    distance: torch.Tensor,
    closing_speed: torch.Tensor,
    *,
    reach_threshold: float = 0.2,
    movement_weight: float = 3.0,
    movement_fade_distance: float = 0.3,
    movement_speed_limit: float = 1.0,
) -> torch.Tensor:
    """Legacy blocker-speed reward for retrospective comparisons, not live shaping.

    Closing speed is -d(distance)/dt for the current nearest blocker, not a
    region-selected torso velocity. No history or episode state is needed.
    """
    if distance.ndim != 1 or closing_speed.shape != distance.shape:
        raise ValueError("Distance and closing speed must have matching (B,) shapes")
    for name, value in (("reach_threshold", reach_threshold), ("movement_weight", movement_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (
        ("movement_fade_distance", movement_fade_distance),
        ("movement_speed_limit", movement_speed_limit),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    gate = ((distance - reach_threshold) / movement_fade_distance).clamp(0, 1)
    speed = closing_speed.clamp(-movement_speed_limit, movement_speed_limit)
    return movement_weight * gate * speed


def movement_guided_intercept_reward(
    distance: torch.Tensor,
    active_mask: torch.Tensor,
    closing_speed: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    *,
    distance_sigma: float = 5.0,
    reach_threshold: float = 0.2,
    movement_weight: float = 3.0,
    movement_fade_distance: float = 0.3,
    movement_speed_limit: float = 1.0,
) -> torch.Tensor:
    """Legacy combined formula; live adapters use timed coverage and root progress."""
    proximity = intercept_proximity_reward(
        distance, active_mask,
        distance_sigma=distance_sigma, reach_threshold=reach_threshold,
    )
    movement = intercept_movement_reward(
        distance, closing_speed,
        reach_threshold=reach_threshold, movement_weight=movement_weight,
        movement_fade_distance=movement_fade_distance,
        movement_speed_limit=movement_speed_limit,
    )
    reward = proximity + movement
    upright = 1.0 - projected_gravity_b[:, :2].square().sum(dim=1).clamp(0, 1)
    return torch.where(active_mask, reward * upright, torch.zeros_like(reward))

def ee_reach_from_env(
    env: ManagerBasedRlEnv,
    blocker_cfg: SceneEntityCfg,
    ball_name: str = "ball",
    distance_sigma: float = 5.0,
    reach_threshold: float = 0.2,
    guidance: dict | None = None,
) -> torch.Tensor:
    """Time-weighted blocker coverage; no positive hand-speed bonus."""
    f = interception_features(env, blocker_cfg, ball_name=ball_name, guidance=guidance,
                              reach_threshold=reach_threshold)
    proximity = intercept_proximity_reward(
        f["distance"], f["active"], distance_sigma=distance_sigma, reach_threshold=reach_threshold,
    )
    reward = proximity * f["time_weight"] * f["settling"] * f["upright"]
    return torch.where(f["active"], reward, 0.)


def body_intercept_from_env(
    env: ManagerBasedRlEnv, blocker_cfg: SceneEntityCfg, ball_name: str = "ball",
    guidance: dict | None = None, lateral_reach: float = .3,
    movement_weight: float = 2., fade_distance: float = .3,
    reach_threshold: float = .05, position_gain: float = 1.,
    max_speed: float = .3, velocity_std: float = .2,
) -> torch.Tensor:
    """Active-shot lateral tracking relative to standing; coverage commands a stop."""
    f = interception_features(env, blocker_cfg, ball_name=ball_name, guidance=guidance,
                              reach_threshold=reach_threshold)
    robot = env.scene[blocker_cfg.name]
    reward = lateral_velocity_tracking_reward(
        f["target"][:, 1], robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1],
        robot.data.root_link_lin_vel_w[:, 1], f["distance"], lateral_reach=lateral_reach,
        movement_weight=movement_weight, fade_distance=fade_distance,
        reach_threshold=reach_threshold, position_gain=position_gain,
        max_speed=max_speed, velocity_std=velocity_std,
    )
    return torch.where(f["active"], reward * f["upright"], 0.)

def stopball_from_env(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    state = get_shot_outcome_state(env)

    return _active_shot_reward(
        env,
        state.stopball_reward,
    )

def target_success_reward(
    distance: torch.Tensor,
    success_flag: torch.Tensor,
    strict_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward target proximity, not physical contact or a final save, shape (B,)."""
    if (
        success_flag.dtype != torch.bool
        or success_flag.shape != distance.shape
    ):
        raise ValueError("success_flag must be a boolean tensor of the same shape as distance")

    close_to_target = distance < strict_threshold

    return (1.0 + success_flag.float()) * close_to_target.float()

def target_success_from_env(
    env: ManagerBasedRlEnv,
    blocker_cfg: SceneEntityCfg,
    strict_threshold: float = 0.15,
    guidance: dict | None = None,
) -> torch.Tensor:
    """Time-aligned attainment of the same corrected target as reach shaping."""
    f = interception_features(env, blocker_cfg, guidance=guidance)
    reward = (f["distance"] < strict_threshold).float() * f["time_weight"] * f["settling"]
    return torch.where(f["active"], reward, 0.)

def _actuated_joint_ids(robot) -> torch.Tensor:
    groups = []

    for actuator in robot.actuators:
        if actuator.cfg.transmission_type != TransmissionType.JOINT:
            raise ValueError("Only joint actuators are supported")
        groups.append(actuator.target_ids)

    if not groups:
        raise ValueError("No actuators found")

    joint_ids = torch.cat(groups)

    if joint_ids.numel() == 0:
        raise ValueError("No actuated joints found")

    if torch.unique(joint_ids).numel() != joint_ids.numel():
        raise ValueError("Multiple actuators target the same joint")

    return joint_ids

def _masked_recovery_exp(
    error: torch.Tensor,
    recovery_mask: torch.Tensor,
    coefficient: float = 5.0,
) -> torch.Tensor:
    """Exponential reward for recovery, shape (B,)"""
    if recovery_mask.dtype != torch.bool:
        raise ValueError("Recovery mask must be a boolean tensor")

    if recovery_mask.shape != error.shape:
        raise ValueError("Recovery mask and error must have the same shape")

    reward = torch.exp(-coefficient * error)

    return torch.where(
        recovery_mask,
        reward,
        torch.zeros_like(reward)
    )

def _check_foot_vectors(value: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[1:] != (2, 3):
        raise ValueError("Foot vectors must have shape (B, 2, 3)")

def _active_shot_reward(
    env: ManagerBasedRlEnv,
    reward: torch.Tensor,
) -> torch.Tensor:
    resolved = env._goalkeeper_outcome_state.shot_resolved

    return torch.where(
        resolved,
        torch.zeros_like(reward),
        reward
    )
