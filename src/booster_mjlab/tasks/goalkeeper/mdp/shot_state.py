from __future__ import annotations

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
    ):
        self.ball_name = ball_name
        self.velocity_change_threshold = velocity_change_threshold

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

    outcome = env._goalkeeper_outcome_state

    return (
        behind_reference
        | velocity_changed
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''检测球是否停止或成功接触, shape (B,)'''
    for flag in (stop_flag, success_flag):
        if (
            flag.dtype != torch.bool
            or flag.shape != ball_position_x_e.shape
        ):
            raise ValueError("Flags must be boolean with shape (B,)")
        
    changed_velocity = (
        (
            ball_velocity_x_w - launch_velocity_x_w
            > velocity_change_threshold
        )
        & (ball_position_x_e > 0.0)
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
):
    if not hasattr(env, "_goalkeeper_shot_state"):
        env._goalkeeper_shot_state = GoalKeeperShotState(
            num_envs=env.num_envs,
            device=env.device,
            ball_name=ball_name,
            velocity_change_threshold=velocity_change_threshold,
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