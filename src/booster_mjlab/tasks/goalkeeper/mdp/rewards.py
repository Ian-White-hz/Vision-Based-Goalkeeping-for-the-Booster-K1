from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab.utils.lab_api.math import quat_apply_inverse

from .shot_state import (
    get_recovery_mask,
    get_hand_target_distance,
    stopball_transition,
    get_shot_outcome_state
)

import math
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

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

def ee_reach_reward(
    distance: torch.Tensor,
    target_position_e: torch.Tensor,
    torso_position_e: torch.Tensor,
    body_velocity_w: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    ball_position_x_e: torch.Tensor,
    ball_velocity_x_w: torch.Tensor,
    launch_velocity_x_w: torch.Tensor,
    region: torch.Tensor,
    *,
    curriculum_level: float = 0.0,
    distance_sigma: float = 5.0,
    reach_threshold: float = 0.2,
) -> torch.Tensor:
    """奖励机器人接近目标位置, shape (B,)"""
    delta_vx = ball_velocity_x_w - launch_velocity_x_w

    phase1 = (
        (ball_position_x_e > 1.5)
        & (delta_vx < 2.0)
    )
    recovery = (
        (ball_position_x_e < 0.0)
        | (delta_vx > 2.0)
    )

    # 远球阶段：身体横向对齐、根据目标高度调整高度。
    lateral = (
        target_position_e[:, 1] - torso_position_e[:, 1]
    ).clamp(-1.0, 1.0)

    lateral = torch.where(
        lateral.abs() < 0.3,
        torch.zeros_like(lateral),
        lateral,
    )

    vertical = (
        torso_position_e[:, 2]
        - target_position_e[:, 2].clamp(0.3, 1.2)
    ).clamp(0.0, 1.0)

    alignment_reward = 1.0 - (
        vertical + lateral.abs()
    ) / 2.0

    # 近球阶段：按区域鼓励对应方向的身体运动。
    left_region = (region == 0) | (region == 4)
    right_region = (region == 1) | (region == 5)
    jump_region = (region == 2) | (region == 3)

    velocity_scale = torch.zeros_like(distance)
    velocity_scale = torch.where(
        left_region,
        1.0 + 3.0 * body_velocity_w[:, 1].clamp(0.0, 3.0),
        velocity_scale,
    )
    velocity_scale = torch.where(
        right_region,
        1.0 - 3.0 * body_velocity_w[:, 1].clamp(-3.0, 0.0),
        velocity_scale,
    )

    jump_scale = 3.0 + 3.0 * curriculum_level
    velocity_scale = torch.where(
        jump_region,
        1.0 + jump_scale * body_velocity_w[:, 2].clamp(0.0, 3.0),
        velocity_scale,
    )
    velocity_scale = torch.where(
        recovery,
        torch.full_like(velocity_scale, 2.0),
        velocity_scale,
    )

    # 与原项目的逻辑等价，但 sigmoid 写法数值更稳定。
    distance_reward = torch.sigmoid(
        -distance_sigma * (distance - reach_threshold)
    )
    reach_reward = distance_reward * velocity_scale

    reward = torch.where(
        phase1,
        alignment_reward,
        reach_reward,
    )

    upright = 1.0 - projected_gravity_b[:, :2].square().sum(
        dim=1
    ).clamp(0.0, 1.0)

    return reward * upright

def ee_reach_from_env(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    ball_name: str = "ball",
    curriculum_level: float = 0.0,
    distance_sigma: float = 5.0,
    reach_threshold: float = 0.2,
) -> torch.Tensor:
    distance = get_hand_target_distance(env, hand_cfg)

    robot = env.scene[hand_cfg.name]
    ball = env.scene[ball_name]
    shots = env._full_goal_shots
    origins = env.scene.env_origins

    reward = ee_reach_reward(
        distance=distance,
        target_position_e=shots["target_position_e"],
        torso_position_e=(
            robot.data.root_link_pos_w - origins
        ),
        body_velocity_w=robot.data.root_link_lin_vel_w,
        projected_gravity_b=robot.data.projected_gravity_b,
        ball_position_x_e=(
            ball.data.root_link_pos_w[:, 0] - origins[:, 0]
        ),
        ball_velocity_x_w=ball.data.root_link_lin_vel_w[:, 0],
        launch_velocity_x_w=shots["velocity"][:, 0],
        region=shots["region"],
        curriculum_level=curriculum_level,
        distance_sigma=distance_sigma,
        reach_threshold=reach_threshold,
    )

    return _active_shot_reward(env, reward)

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
    """奖励机器人成功接触目标, shape (B,)"""
    if (
        success_flag.dtype != torch.bool
        or success_flag.shape != distance.shape
    ):
        raise ValueError("success_flag must be a boolean tensor of the same shape as distance")

    close_to_target = distance < strict_threshold

    return (1.0 + success_flag.float()) * close_to_target.float()

def target_success_from_env(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    strict_threshold: float = 0.15,
) -> torch.Tensor:
    distance = get_hand_target_distance(env, hand_cfg)
    state = get_shot_outcome_state(env)

    reward = target_success_reward(
        distance=distance,
        success_flag=state.success_before_step,
        strict_threshold=strict_threshold,
    )

    return _active_shot_reward(env, reward)

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