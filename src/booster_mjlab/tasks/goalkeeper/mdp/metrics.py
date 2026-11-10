"""Behavior metrics shared by PPO and AMP goalkeeper tasks."""

import torch

from booster_mjlab.tasks.goalkeeper.amp_interfaces import get_amp_motion_ids

from .outcomes import update_goalkeeper_outcome_state
from .shot_state import get_shot_outcome_state
from .terminations import excessive_body_tilt


def category_coverage(env, motion_id: int) -> torch.Tensor:
    return (get_amp_motion_ids(env) == motion_id).to(torch.float32)


def category_save(env, motion_id: int) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    return (category & update_goalkeeper_outcome_state(env).saved).to(torch.float32)


def category_contact(env, motion_id: int) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    return (category & update_goalkeeper_outcome_state(env).robot_contact_seen).to(
        torch.float32
    )


def category_velocity_change(env, motion_id: int) -> torch.Tensor:
    """Legacy stopball success flag, explicitly a velocity-change proxy."""
    category = get_amp_motion_ids(env) == motion_id
    return (category & get_shot_outcome_state(env).success_flag).to(torch.float32)


def category_miss(env, motion_id: int) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    state = update_goalkeeper_outcome_state(env)
    return (category & state.shot_resolved & ~state.goal_conceded & ~state.saved).to(
        torch.float32
    )


def category_unresolved(env, motion_id: int) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    state = update_goalkeeper_outcome_state(env)
    return (category & ~state.shot_resolved & ~state.saved).to(torch.float32)


def category_goal(env, motion_id: int) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    return (category & update_goalkeeper_outcome_state(env).goal_conceded).to(
        torch.float32
    )


def category_fall(
    env,
    motion_id: int,
    robot_name: str = "robot",
    max_tilt_deg: float = 60.0,
) -> torch.Tensor:
    category = get_amp_motion_ids(env) == motion_id
    fallen = excessive_body_tilt(env, robot_name=robot_name, max_tilt_deg=max_tilt_deg)
    return (category & fallen).to(torch.float32)
