import math

import torch

from .shot_state import robot_ball_contact_this_step


class GoalkeeperOutcomeState:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        goal_x: float,
        goal_width: float,
        goal_height: float,
        ball_radius: float,
        ball_x_bounds: tuple[float, float],
        ball_y_limit: float,
        ground_tolerance: float = 0.01,
        stop_speed: float = 0.12,
        stop_hold_s: float = 0.4,
        safe_stop_x: float = 0.1,
    ):
        if (
            not math.isfinite(stop_speed) or stop_speed <= 0.0
            or not math.isfinite(stop_hold_s) or stop_hold_s <= 0.0
            or not math.isfinite(safe_stop_x)
        ):
            raise ValueError("Invalid safe-stop parameters")
        self.stop_speed = stop_speed
        self.stop_hold_s = stop_hold_s
        self.safe_stop_x = safe_stop_x
        self.goal_parameters = {
            "goal_x": goal_x,
            "goal_width": goal_width,
            "goal_height": goal_height,
            "ball_radius": ball_radius,
            "ground_tolerance": ground_tolerance,
        }

        self.previous_ball_position_e = torch.zeros(
            (num_envs, 3),
            device=device,
        )

        self.goal_conceded = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.goal_conceded_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )

        self.boundary_parameters = {
            "x_bounds": ball_x_bounds,
            "y_limit": ball_y_limit,
            "ball_radius": ball_radius,
        }

        self.ball_out_of_bounds = torch.zeros_like(self.goal_conceded)
        self.shot_resolved = torch.zeros_like(self.goal_conceded)

        self.shot_resolved_step = torch.full_like(
            self.goal_conceded_step,
            -1,
        )
        self.robot_contact_seen = torch.zeros_like(self.goal_conceded)
        self.cleared = torch.zeros_like(self.goal_conceded)
        self.safe_stopped = torch.zeros_like(self.goal_conceded)
        self.saved = torch.zeros_like(self.goal_conceded)
        self.safe_stop_steps = torch.zeros_like(self.goal_conceded_step)
        self.last_update_step = torch.full_like(self.goal_conceded_step, -1)
        self._last_adapter_update_step: int | None = None

    def reset(
        self,
        ball_position_e: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        ids = slice(None) if env_ids is None else env_ids

        self.previous_ball_position_e[ids] = ball_position_e[ids]

        for value in (
            self.goal_conceded,
            self.ball_out_of_bounds,
            self.shot_resolved,
            self.robot_contact_seen,
            self.cleared,
            self.safe_stopped,
            self.saved,
            self.safe_stop_steps,
        ):
            value[ids] = False

        self.goal_conceded_step[ids] = -1
        self.shot_resolved_step[ids] = -1
        self.last_update_step[ids] = -1
        self._last_adapter_update_step = None

    def update(
        self,
        ball_position_e: torch.Tensor,
        step: int,
        *,
        ball_velocity_w: torch.Tensor | None = None,
        robot_ball_contact: torch.Tensor | None = None,
        step_dt: float | None = None,
    ):
        fresh = self.last_update_step != step
        if not fresh.any():
            return
        if robot_ball_contact is not None:
            if (
                robot_ball_contact.shape != self.robot_contact_seen.shape
                or robot_ball_contact.dtype != torch.bool
                or robot_ball_contact.device != self.robot_contact_seen.device
            ):
                raise ValueError("robot_ball_contact must be bool (num_envs,)")
            self.robot_contact_seen |= fresh & robot_ball_contact

        crossed = goal_crossing_mask(
            self.previous_ball_position_e,
            ball_position_e,
            **self.goal_parameters,
        )

        outside = ball_out_of_bounds_mask(
            ball_position_e,
            **self.boundary_parameters,
        )

        active = fresh & ~self.shot_resolved

        first_goal = active & crossed
        first_out = active & outside & ~crossed
        newly_resolved = first_goal | first_out

        self.goal_conceded_step[first_goal] = step
        self.shot_resolved_step[newly_resolved] = step

        self.goal_conceded |= first_goal
        self.ball_out_of_bounds |= first_out
        self.shot_resolved |= newly_resolved
        # Only a contact occurring by the out-of-bounds decision can earn a
        # clearance. A later incidental contact cannot relabel an unassisted miss.
        self.cleared |= first_out & self.robot_contact_seen

        stopped = torch.zeros_like(self.safe_stopped)
        if ball_velocity_w is not None:
            if ball_velocity_w.shape != ball_position_e.shape:
                raise ValueError("Ball velocity and position shapes must match")
            if step_dt is None or not math.isfinite(step_dt) or step_dt <= 0.0:
                raise ValueError("Safe-stop evaluation requires positive finite step_dt")
            stopped = (
                self.robot_contact_seen
                & ~self.shot_resolved
                & torch.isfinite(ball_position_e).all(dim=-1)
                & torch.isfinite(ball_velocity_w).all(dim=-1)
                & (ball_position_e[:, 0] >= self.safe_stop_x)
                & (ball_velocity_w.norm(dim=-1) <= self.stop_speed)
            )
        next_count = torch.where(stopped, self.safe_stop_steps + 1, 0)
        self.safe_stop_steps.copy_(torch.where(fresh, next_count, self.safe_stop_steps))
        hold_steps = math.ceil(self.stop_hold_s / step_dt) if ball_velocity_w is not None else 1
        safe_stopped = stopped & (self.safe_stop_steps >= hold_steps)
        self.safe_stopped.copy_(torch.where(fresh, safe_stopped, self.safe_stopped))
        # A held ball is only safe while the hold still applies. It does not
        # end the episode early; if it moves and later concedes, save becomes false.
        self.saved.copy_((self.cleared | self.safe_stopped) & ~self.goal_conceded)
        self.previous_ball_position_e.copy_(
            torch.where(fresh[:, None], ball_position_e, self.previous_ball_position_e)
        )
        self.last_update_step[fresh] = step


def goal_crossing_mask(
    previous_ball_position_e: torch.Tensor,
    ball_position_e: torch.Tensor,
    *,
    goal_x: float,
    goal_width: float,
    goal_height: float,
    ball_radius: float,
    ground_tolerance: float = 0.01,
) -> torch.Tensor:
    """本步是否向负 x 方向完整穿过球门，返回 shape (B,) 的 bool。"""
    previous = previous_ball_position_e
    current = ball_position_e

    # 球心到达这个 x 位置时，球的最后一侧刚好到达门线。
    crossing_x = goal_x - ball_radius

    # 检查球是否从上一步到这一步穿过了门线。
    crossed = (
        (previous[:, 0] >= crossing_x)
        & (current[:, 0] < crossing_x)
    )

    displacement = current - previous
    delta_x = displacement[:, 0]

    # 未移动的球不可能过线；这里仅避免插值计算除以零。
    safe_delta_x = torch.where(
        delta_x != 0.0,
        delta_x,
        torch.ones_like(delta_x),
    )

    alpha = (
        (crossing_x - previous[:, 0]) / safe_delta_x
    ).clamp(0.0, 1.0)

    crossing_position = previous + alpha[:, None] * displacement

    inside_width = (
        crossing_position[:, 1].abs() + ball_radius
        <= goal_width / 2.0
    )

    inside_height = (
        (crossing_position[:, 2] >= ball_radius - ground_tolerance)
        & (crossing_position[:, 2] + ball_radius <= goal_height)
    )

    finite = (
        torch.isfinite(previous).all(dim=-1)
        & torch.isfinite(current).all(dim=-1)
    )

    return crossed & inside_width & inside_height & finite

def ball_out_of_bounds_mask(
    ball_position_e: torch.Tensor,
    *,
    x_bounds: tuple[float, float],
    y_limit: float,
    ball_radius: float,
) -> torch.Tensor:
    """球是否已经完全出界，返回 shape (B,) 的 bool。"""
    x_min, x_max = x_bounds

    if x_min >= x_max or y_limit <= 0.0 or ball_radius <= 0.0:
        raise ValueError("Invalid ball boundary parameters")

    x = ball_position_e[:, 0]
    y = ball_position_e[:, 1]

    outside = (
        (x + ball_radius < x_min)
        | (x - ball_radius > x_max)
        | (y.abs() - ball_radius > y_limit)
    )

    finite = torch.isfinite(ball_position_e).all(dim=-1)

    return outside & finite

def reset_goalkeeper_outcome_state(
    env,
    env_ids,
    settings,
):
    state = getattr(env, "_goalkeeper_outcome_state", None)

    if state is None:
        state = GoalkeeperOutcomeState(
            num_envs=env.num_envs,
            device=env.device,
            goal_x=settings.goal_x,
            goal_width=settings.goal_width,
            goal_height=settings.goal_height,
            ball_radius=settings.ball_radius,
            ball_x_bounds=settings.ball_x_bounds,
            ball_y_limit=settings.ball_y_limit,
            stop_speed=settings.stop_speed,
            stop_hold_s=settings.stop_hold_s,
            safe_stop_x=settings.intercept_x,
        )
        env._goalkeeper_outcome_state = state

    shots = env._full_goal_shots
    initial_position_e = shots.get("start_position_e")

    if initial_position_e is None:
        # 滚地球模式：与当前 launch_shots 的起点设置一致。
        initial_position_e = torch.stack(
            (
                shots["start_x"],
                shots["start_y"],
                torch.full_like(
                    shots["start_x"],
                    settings.ball_radius + 0.002,
                ),
            ),
            dim=-1,
        )

    state.reset(
        initial_position_e,
        env_ids=env_ids,
    )

def update_goalkeeper_outcome_state(
    env,
    env_ids=None,
    ball_name: str = "ball",
) -> GoalkeeperOutcomeState:
    state = getattr(env, "_goalkeeper_outcome_state", None)

    if state is None:
        raise RuntimeError(
            "GoalkeeperOutcomeState has not been initialized. "
            "Call reset_goalkeeper_outcome_state first."
        )

    # Many category metrics read this state every step. Avoid repeating sensor
    # reductions and device synchronization; a partial reset invalidates this
    # Python-side cache while per-environment freshness still protects other rows.
    if state._last_adapter_update_step == env.common_step_counter:
        return state

    ball_position_e = (
        env.scene[ball_name].data.root_link_pos_w
        - env.scene.env_origins
    )

    ball = env.scene[ball_name]
    state.update(
        ball_position_e=ball_position_e,
        step=env.common_step_counter,
        ball_velocity_w=ball.data.root_link_lin_vel_w,
        robot_ball_contact=robot_ball_contact_this_step(env),
        step_dt=env.step_dt,
    )
    state._last_adapter_update_step = env.common_step_counter

    return state
