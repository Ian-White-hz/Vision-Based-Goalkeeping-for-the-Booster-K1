"""Independent origin/goal sampling and reset-only, pure-rolling launches."""

import torch

from .settings import FullGoalSettings


def sample_shots(count, settings: FullGoalSettings, device="cpu", generator=None):
    if count < 0:
        raise ValueError("count must be nonnegative")

    def uniform(low, high):
        return low + (high - low) * torch.rand(count, device=device, generator=generator)

    x = uniform(*settings.start_x_range)
    y = uniform(*settings.start_y_range)
    target_y = uniform(-settings.target_half_width, settings.target_half_width)
    speed = uniform(*settings.speed_range)
    direction = torch.stack((settings.goal_x - x, target_y - y), dim=1)
    velocity = direction / direction.norm(dim=1, keepdim=True) * speed[:, None]
    return dict(start_x=x, start_y=y, target_y=target_y, speed=speed, velocity=velocity)


def launch_shots(env, env_ids, settings: FullGoalSettings):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    sample = sample_shots(len(ids), settings, device=env.device)
    # Diagnostics only: never automatically inserted into observations.
    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {
            name: torch.zeros((env.num_envs, *value.shape[1:]), device=env.device)
            for name, value in sample.items()
        }
    for name, value in sample.items():
        env._full_goal_shots[name][ids] = value
    pose = torch.zeros((len(ids), 7), device=env.device)
    velocity = torch.zeros((len(ids), 6), device=env.device)
    pose[:, 0], pose[:, 1] = sample["start_x"], sample["start_y"]
    pose[:, 2], pose[:, 3] = settings.ball_radius + 0.002, 1.0
    pose[:, :3] += env.scene.env_origins[ids]
    velocity[:, :2] = sample["velocity"]
    velocity[:, 3] = -velocity[:, 1] / settings.ball_radius
    velocity[:, 4] = velocity[:, 0] / settings.ball_radius
    env.scene["ball"].write_root_state_to_sim(torch.cat((pose, velocity), dim=1), env_ids=ids)
