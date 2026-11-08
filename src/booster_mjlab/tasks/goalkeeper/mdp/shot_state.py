from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

class GoalKeeperShotState:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str | None = None,
        ball_name: str = "ball",
        velocity_change_threshold: float = 2.0,
        rolling_velocity_change_fraction: float = 0.8,
        contact_sensor_name: str = "robot_ball_contact",
    ):
        if (
            not math.isfinite(rolling_velocity_change_fraction)
            or not 0.0 < rolling_velocity_change_fraction <= 1.0
        ):
            raise ValueError("rolling_velocity_change_fraction must be in (0, 1]")
        self.ball_name = ball_name
        self.velocity_change_threshold = velocity_change_threshold
        self.rolling_velocity_change_fraction = rolling_velocity_change_fraction
        self.contact_sensor_name = contact_sensor_name

        self.stop_flag = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        ) 
        self.success_flag = torch.zeros_like(self.stop_flag)
        self.success_before_step = torch.zeros_like(self.stop_flag)

        self.stopball_reward = torch.zeros(
            num_envs, device=device,
        )
        self.last_update_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        for value in (
            self.stop_flag,
            self.success_flag,
            self.success_before_step,
            self.stopball_reward,
        ):
            value[env_ids] = 0

        self.last_update_step[env_ids] = -1

    def update(self, env):
        ball = env.scene[self.ball_name]
        shots = env._full_goal_shots

        fresh = self.last_update_step != env.common_step_counter

        is_rolling = shots.get("is_rolling", torch.zeros_like(self.stop_flag))
        robot_ball_contact = torch.zeros_like(self.stop_flag)
        if is_rolling.any():
            try:
                contact = env.scene[self.contact_sensor_name].data
            except KeyError as error:
                raise RuntimeError(
                    "Rolling shot success requires the robot-ball contact sensor "
                    f"{self.contact_sensor_name!r}"
                ) from error
            if contact.found is None or contact.force_history is None:
                raise RuntimeError(
                    "Robot-ball contact sensor requires found and force history"
                )
            # A collision may end before the final physics substep. Only keep
            # contact evidence from this control step, not the whole episode:
            # a light touch followed much later by friction is not a save.
            robot_ball_contact = (contact.found > 0).any(dim=1) | (
                contact.force_history.square().sum(dim=-1) > 1.0e-12
            ).any(dim=(1, 2))

        reward, next_stop, next_success = stopball_transition(
            ball_position_x_e=(
                ball.data.root_link_pos_w[:, 0]
                - env.scene.env_origins[:, 0]
            ),
            ball_velocity_x_w=ball.data.root_link_lin_vel_w[:, 0],
            launch_velocity_x_w=shots["velocity"][:, 0],
            stop_flag=self.stop_flag,
            success_flag=self.success_flag,
            velocity_change_threshold=self.velocity_change_threshold,
            is_rolling=is_rolling,
            robot_ball_contact=robot_ball_contact,
            rolling_velocity_change_fraction=self.rolling_velocity_change_fraction,
        )

        self.success_before_step.copy_(
            torch.where(
                fresh, self.success_flag, self.success_before_step
            )
        )
        self.stopball_reward.copy_(
            torch.where(fresh, reward, self.stopball_reward)
        )
        self.stop_flag.copy_(
            torch.where(fresh, next_stop, self.stop_flag)
        )
        self.success_flag.copy_(
            torch.where(fresh, next_success, self.success_flag)
        )
        self.last_update_step.fill_(env.common_step_counter)

def get_recovery_mask(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    reference_x: float = 0.0,
    velocity_change_threshold: float = 2.0,
) -> torch.Tensor:
    '''获取机器人恢复掩码, shape (B,)'''
    shots = getattr(env, "_full_goal_shots", None)
    if shots is None:
        raise RuntimeError("Full goal shots not found in env. Make sure to call `launch_shots` before calling this function.")

    ball = env.scene[ball_name]

    ball_x_e = (
        ball.data.root_link_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )

    # 机器人当前 x 方向速度
    current_velocity_x = ball.data.root_link_lin_vel_w[:, 0]
    # 机器人发射时 x 方向速度
    launch_velocity_x = shots["velocity"][:, 0]

    # 机器人是否在参考位置后方
    behind_reference = ball_x_e < reference_x
    # 机器人当前速度是否比发射时速度增加了阈值，因为发射时速度是负的，所以增加意味着变得更小。
    velocity_changed = (
        current_velocity_x - launch_velocity_x
        > velocity_change_threshold
    )
    # Rolling shots use the contact-gated relative criterion in the shared
    # shot state, never a second, contact-free velocity threshold here.
    is_rolling = shots.get("is_rolling", torch.zeros_like(behind_reference))
    velocity_changed &= ~is_rolling
    saved = torch.zeros_like(behind_reference)
    shot_state = getattr(env, "_goalkeeper_shot_state", None)
    if shot_state is not None:
        shot_state.update(env)
        saved = shot_state.success_flag

    outcome = env._goalkeeper_outcome_state

    return (
        behind_reference
        | velocity_changed
        | saved
        | outcome.shot_resolved
    )

def compute_rolling_shot_target(
    start_x: torch.Tensor,
    start_y: torch.Tensor,
    goal_y: torch.Tensor,
    *,
    goal_x: torch.Tensor,
    ball_radius: float,
    intercept_x: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    fraction = (
        (intercept_x - start_x) / (goal_x - start_x)
    )

    intercept_y = start_y + fraction * (goal_y - start_y)

    target_position_e = torch.stack(
        (
            torch.full_like(start_x, intercept_x),
            intercept_y,
            torch.full_like(start_x, ball_radius),
        ),
        dim=-1
    )

    region = torch.where(intercept_y >= 0.0, 4, 5)

    return target_position_e, region

def get_hand_target_distance(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    '''获取手与目标的距离, shape (B,)'''
    shots = getattr(env, "_full_goal_shots", None)
    if shots is None:
        raise RuntimeError("Full goal shots not found in env. Make sure to call `launch_shots` before calling this function.")

    robot = env.scene[hand_cfg.name]

    hand_position_w = robot.data.site_pos_w[
        :, hand_cfg.site_ids, :
    ]
    if hand_position_w.shape[1] != 2:
        raise ValueError(f"Expected 2 sites for hand, got {hand_position_w.shape[1]}")
    # 
    hand_position_e = (
        hand_position_w
        - env.scene.env_origins[:, None, :]
    )

    region = shots["region"]
    use_left_hand = (
        (region == 0)
        | (region == 2)
        | (region == 4)
    )

    selected_hand_e = torch.where(
        use_left_hand[:, None],
        hand_position_e[:, 0, :],
        hand_position_e[:, 1, :],
    )

    return torch.linalg.vector_norm(
        selected_hand_e - shots["target_position_e"],
        dim=-1
    )

def stopball_transition(
    ball_position_x_e: torch.Tensor,
    ball_velocity_x_w: torch.Tensor,
    launch_velocity_x_w: torch.Tensor,
    stop_flag: torch.Tensor,
    success_flag: torch.Tensor,
    velocity_change_threshold: float = 2.0,
    *,
    is_rolling: torch.Tensor | None = None,
    robot_ball_contact: torch.Tensor | None = None,
    rolling_velocity_change_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot save proxy; rolling saves require contact in this control step.

    Ballistic shots retain the absolute delta-v criterion. Rolling shots use
    a fraction of the incoming x speed, so both stopping and reflecting a slow
    ball can count. This is not a final no-goal outcome classification.
    """
    for flag in (stop_flag, success_flag):
        if (
            flag.dtype != torch.bool
            or flag.shape != ball_position_x_e.shape
        ):
            raise ValueError("Flags must be boolean with shape (B,)")
    if (
        not math.isfinite(rolling_velocity_change_fraction)
        or not 0.0 < rolling_velocity_change_fraction <= 1.0
    ):
        raise ValueError("rolling_velocity_change_fraction must be in (0, 1]")
    if is_rolling is None:
        is_rolling = torch.zeros_like(stop_flag)
    if robot_ball_contact is None:
        if is_rolling.any():
            raise ValueError("Rolling shots require robot_ball_contact")
        robot_ball_contact = torch.zeros_like(stop_flag)
    for flag in (is_rolling, robot_ball_contact):
        if flag.dtype != torch.bool or flag.shape != ball_position_x_e.shape:
            raise ValueError("Shot masks must be boolean with shape (B,)")

    delta_vx = ball_velocity_x_w - launch_velocity_x_w
    rolling_changed = (
        (delta_vx >= rolling_velocity_change_fraction * (-launch_velocity_x_w))
        & robot_ball_contact
    )
    changed_velocity = torch.where(
        is_rolling,
        rolling_changed,
        delta_vx > velocity_change_threshold,
    )
    changed_velocity &= (
        (ball_position_x_e > 0.0)
        & (launch_velocity_x_w < -1.0e-6)
        & torch.isfinite(ball_position_x_e)
        & torch.isfinite(ball_velocity_x_w)
        & torch.isfinite(launch_velocity_x_w)
    )

    first_event = (~stop_flag) & changed_velocity
    reward = first_event.float()

    next_stop_flag = (
        stop_flag
        | changed_velocity
        | (ball_position_x_e < 0.0)
    )
    next_success_flag = success_flag | first_event

    return reward, next_stop_flag, next_success_flag

def reset_shot_outcome_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    ball_name: str = "ball",
    velocity_change_threshold: float = 2.0,
    rolling_velocity_change_fraction: float = 0.8,
    contact_sensor_name: str = "robot_ball_contact",
):
    if not hasattr(env, "_goalkeeper_shot_state"):
        env._goalkeeper_shot_state = GoalKeeperShotState(
            num_envs=env.num_envs,
            device=env.device,
            ball_name=ball_name,
            velocity_change_threshold=velocity_change_threshold,
            rolling_velocity_change_fraction=rolling_velocity_change_fraction,
            contact_sensor_name=contact_sensor_name,
        )

    env._goalkeeper_shot_state.reset(env_ids=env_ids)

def get_shot_outcome_state(
    env: ManagerBasedRlEnv,
) -> GoalKeeperShotState:
    state = getattr(env, "_goalkeeper_shot_state", None)

    if state is None:
        raise RuntimeError(
            "GoalKeeperShotState not found in env. Make sure to call `reset_shot_outcome_state` before calling this function."
        )

    state.update(env)
    return state
