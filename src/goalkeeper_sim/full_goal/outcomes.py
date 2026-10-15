"""Rolling-shot outcomes, separate from policy observations and rewards."""

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase


def goal_crossed(position, settings):
    return ((position[:, 0] + settings.ball_radius < settings.goal_x)
            & (position[:, 1].abs() < settings.goal_width / 2 + settings.ball_radius))


def valid_stop(touched, ground, support, goal, failed, speed, ball_x, distance, settings):
    return (touched & ground & support & ~goal & ~failed
            & (speed < settings.stop_speed)
            & (ball_x - settings.ball_radius > settings.goal_x)
            & (distance < settings.stop_distance))


def fallen(env):
    robot = env.scene["robot"].data
    return ((robot.projected_gravity_b[:, 2] > -0.85)
            | (robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < 0.36))


def out_of_guard_region(env, settings):
    position = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return (position[:, 0].abs() > settings.robot_x_limit) | (position[:, 1].abs() > settings.robot_y_limit)


def invalid_ball(env):
    ball = env.scene["ball"].data
    return (~torch.isfinite(ball.root_link_pos_w).all(dim=1)
            | ~torch.isfinite(ball.root_link_lin_vel_w).all(dim=1)
            | (ball.root_link_lin_vel_w.norm(dim=1) > 15))


class FullGoalOutcome(ManagerTermBase):
    """Cache events before auto-reset; a timeout is not defensive success."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.settings = cfg.params["settings"]
        self.required_hold_steps = max(1, math.ceil(self.settings.stop_hold_s / env.step_dt - 1e-9))
        self.goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.touched = self.goal.clone()
        self.stopped = self.goal.clone()
        self.failed = self.goal.clone()
        self.hold = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stamp = torch.full_like(self.hold, -1)
        self.last = {name: self.goal.clone() for name in ("goal", "touched", "stopped", "failed")}
        env._full_goal_outcome = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name, last in self.last.items():
            last[ids] = getattr(self, name)[ids]
            getattr(self, name)[ids] = False
        self.hold[ids], self.stamp[ids] = 0, -1

    def update(self, env):
        fresh = self.stamp != env.common_step_counter
        robot, ball = env.scene["robot"].data, env.scene["ball"].data
        position = ball.root_link_pos_w - env.scene.env_origins
        illegal = env.scene["non_foot_ground_contact"].data.found.flatten(1).any(dim=1)
        self.failed[fresh] |= (fallen(env) | out_of_guard_region(env, self.settings) | illegal | invalid_ball(env))[fresh]
        self.goal[fresh] |= goal_crossed(position, self.settings)[fresh]
        support = ((robot.projected_gravity_b[:, 2] < -0.90)
                   & (robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] > 0.40)
                   & env.scene["feet_ground_contact"].data.found.flatten(1).any(dim=1))
        contact = env.scene["body_ball_contact"].data.found.flatten(1).any(dim=1)
        self.touched[fresh] |= (contact & support & ~self.failed & ~self.goal)[fresh]
        ground = env.scene["ball_ground_contact"].data.found.flatten(1).any(dim=1)
        stopped = valid_stop(self.touched, ground, support, self.goal, self.failed,
                             ball.root_link_lin_vel_w[:, :2].norm(dim=1), position[:, 0],
                             (ball.root_link_pos_w - robot.root_link_pos_w)[:, :2].norm(dim=1), self.settings)
        self.hold[fresh] = torch.where(stopped[fresh], self.hold[fresh] + 1, 0)
        self.stopped[fresh] |= self.hold[fresh] >= self.required_hold_steps
        self.stamp[fresh] = env.common_step_counter

    def __call__(self, env, settings):
        self.update(env)
        return self.goal


def ball_stopped(env):
    outcome = env._full_goal_outcome
    outcome.update(env)
    return outcome.stopped & ~outcome.failed & ~outcome.goal
