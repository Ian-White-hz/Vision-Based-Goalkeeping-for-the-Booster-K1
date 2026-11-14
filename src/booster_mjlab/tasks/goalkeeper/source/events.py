"""Source ballistic shots, fixed region assignment, reset and curriculum."""

import random
import torch
from .state import state
from .settings import (
    GOAL_X,
    INTERCEPT_X,
    INTERCEPT_HEIGHTS,
    REGIONS,
    CURRICULUM_EMA,
    SHOT_START_X,
    SHOT_INTERCEPT_TIME,
)


def reset(env, env_ids=None):
    s = state(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not len(ids):
        return
    r = env.scene["robot"]
    if (env.episode_length_buf[ids] > 0).any():
        env.extras.setdefault("log", {})["Source/save_velocity_proxy"] = s.success_flag[
            ids
        ].mean()
        env.extras["log"]["Source/episode_seconds"] = (
            env.episode_length_buf[ids].float().mean() * env.step_dt
        )
    # The original curriculum is based on episode length, not save rate.
    # Upstream reads only the episodes resetting at the update step; with few
    # envs that flips the level (and reward multipliers) between 0 and 3, so
    # use an EMA over all finished episodes instead.
    lengths = env.episode_length_buf[ids].float()
    lengths = lengths[lengths > 0]
    if len(lengths):
        if s.episode_length_ema < 0:
            s.episode_length_ema = lengths.mean().item()
        else:
            keep = (1.0 - CURRICULUM_EMA) ** len(lengths)
            s.episode_length_ema = (
                keep * s.episode_length_ema + (1 - keep) * lengths.mean().item()
            )
    if not s.settings.play and env.common_step_counter - s.last_step_counter > 500:
        s.startstep = 50 - random.randint(3, 10)
        s.curriculumupdate = int(max(s.episode_length_ema, 0.0) / 50.0)
        delta = 0.3 * s.curriculumupdate
        for low, high in ((0, 1), (2, 3)):
            s.command_ranges[:, low] = (s.command_ranges[:, low] - delta).clamp(
                s.command_bound[:, low], s.command_bound[:, high]
            )
            s.command_ranges[:, high] = (s.command_ranges[:, high] + delta).clamp(
                s.command_bound[:, low], s.command_bound[:, high]
            )
        s.last_step_counter = env.common_step_counter
    q = s.default_dof_pos[ids].clone()
    if s.settings.randomize:
        if torch.rand(()).item() > 0.2:
            q = r.data.joint_pos[
                torch.randint(env.num_envs, (len(ids),), device=env.device)
            ][:, s.jids].clone()
        elif not s.settings.play:
            q = q * torch.empty_like(q).uniform_(0.5, 1.5) + torch.empty_like(
                q
            ).uniform_(-0.1, 0.1)
        q = q.clamp(s.dof_pos_limits[:, 0], s.dof_pos_limits[:, 1])
        s.kp_factor[ids] = torch.empty_like(q).uniform_(0.8, 1.2)
        s.kd_factor[ids] = torch.empty_like(q).uniform_(0.8, 1.2)
        s.offset[ids] = torch.empty_like(q).uniform_(-0.01, 0.01) * s.torque_limits
    exempt = [
        i
        for i, n in enumerate(s.robot.joint_names)
        if n in ("Left_Shoulder_Roll", "Right_Shoulder_Roll")
    ]
    for jid in exempt:
        s.offset[ids, s.jids.index(jid)] = 0.0
    s.init_dof_pos[ids] = q
    r.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=s.jids, env_ids=ids)
    root = r.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:] = (
        torch.empty_like(root[:, 7:]).uniform_(-0.3, 0.3)
        if s.settings.randomize
        else 0.0
    )
    r.write_root_state_to_sim(root, env_ids=ids)

    goal_pose = torch.zeros((len(ids), 7), device=env.device)
    goal_pose[:, :3] = env.scene.env_origins[ids]
    goal_pose[:, 3] = 1.0
    env.scene["goal"].write_mocap_pose_to_sim(goal_pose, env_ids=ids)
    start, velocity, target = sample_region_shots(
        s.command_ranges[ids], s.end_regions[ids]
    )
    # Reward target is the actual ballistic crossing at the robot plane.
    t = (INTERCEPT_X - start[:, 0]) / velocity[:, 0]
    intercept = start + velocity * t[:, None]
    intercept[:, 2] -= 0.5 * 9.81 * t.square()
    s.end_target[ids] = intercept + env.scene.env_origins[ids]
    ball = torch.zeros((len(ids), 13), device=env.device)
    ball[:, :3] = start + env.scene.env_origins[ids]
    ball[:, 3] = 1.0
    ball[:, 7:10] = velocity
    env.scene["ball"].write_root_state_to_sim(ball, env_ids=ids)
    s.jump_state.reset(ids, root[:, 2])
    s.ball_vel[ids] = velocity[:, 0]
    s.first_visible[ids] = -1
    s.first_near[ids] = -1
    s.guide_steps[ids] = 0
    s.catchstep[ids] = 50
    s.vanish_step[ids] = torch.randint(0, 30, (len(ids),), device=env.device)
    for name in (
        "stop_flag",
        "success_flag",
        "has_in_air",
        "ball_last",
        "last_actions",
        "last_last_actions",
        "last_dof_vel",
        "actions",
    ):
        getattr(s, name)[ids] = 0
    s.dist[ids] = 5.0
    s.updated_step = env.common_step_counter
    s.obs_step = -1
    # History is retained across resets as in the source implementation.


def perturb(env, env_ids=None):
    s = state(env)
    if not s.settings.randomize:
        return
    ball = env.scene["ball"]
    v = ball.data.root_link_lin_vel_w
    force = torch.zeros_like(v)
    if s.settings.ball_drag:
        force = (
            -0.5 * 1.225 * 0.47 * torch.pi * 0.1**2 * v.norm(dim=-1, keepdim=True) * v
        )
    if s.settings.ball_force_noise:
        force += torch.empty_like(force).uniform_(-0.5, 0.5)
    ball.write_external_wrench_to_sim(force[:, None], torch.zeros_like(force[:, None]))
    if s.settings.ball_velocity_noise and env.common_step_counter % 25 == 0:
        root = torch.cat(
            (ball.data.root_link_pose_w, ball.data.root_link_vel_w), -1
        ).clone()
        root[:, 7:10] += torch.empty_like(v).uniform_(-0.5, 0.5)
        ball.write_root_state_to_sim(root)
    if not s.settings.play and env.common_step_counter % 750 == 0:
        r = env.scene["robot"]
        root = torch.cat((r.data.root_link_pose_w, r.data.root_link_vel_w), -1).clone()
        root[:, 7:9] = torch.empty_like(root[:, 7:9]).uniform_(-1.5, 1.5)
        r.write_root_state_to_sim(root)


def sample_region_shots(bounds, region_ids=None, max_attempts=512):
    """Reject shots violating either goal bounds or robot-plane constraints."""
    n = len(bounds)
    if region_ids is None:
        reference = torch.tensor(REGIONS, device=bounds.device, dtype=bounds.dtype)
        match = (bounds[:, None] == reference[None]).all(-1)
        if not match.any(-1).all():
            raise ValueError("Explicit region_ids required for custom bounds")
        region_ids = match.long().argmax(-1)
    heights = torch.tensor(INTERCEPT_HEIGHTS, device=bounds.device)[region_ids]
    if (
        not torch.isfinite(bounds).all()
        or ((bounds[:, [1, 3]] - bounds[:, [0, 2]]) <= 0.08).any()
    ):
        raise ValueError("Invalid region bounds")
    start = torch.empty((n, 3), device=bounds.device)
    velocity = torch.empty_like(start)
    target = torch.empty_like(start)
    pending = torch.arange(n, device=bounds.device)
    for _ in range(max_attempts):
        if not len(pending):
            return start, velocity, target
        r = torch.rand((len(pending), 6), device=bounds.device)
        b = bounds[pending]
        x0, x1 = SHOT_START_X
        p = torch.stack(
            (x0 + (x1 - x0) * r[:, 0], -1 + 2 * r[:, 1], 0.15 + 0.65 * r[:, 2]), -1
        )
        end = torch.stack(
            (
                torch.full_like(r[:, 0], GOAL_X),
                b[:, 0] + 0.04 + r[:, 3] * (b[:, 1] - b[:, 0] - 0.08),
                b[:, 2] + 0.04 + r[:, 4] * (b[:, 3] - b[:, 2] - 0.08),
            ),
            -1,
        )
        t0, t1 = SHOT_INTERCEPT_TIME
        ti = t0 + (t1 - t0) * r[:, 5]
        vx = (INTERCEPT_X - p[:, 0]) / ti
        total = (GOAL_X - p[:, 0]) / vx
        v = (end - p) / total[:, None]
        v[:, 2] += 0.5 * 9.81 * total
        zi = p[:, 2] + v[:, 2] * ti - 0.5 * 9.81 * ti.square()
        # Integration margin for dt=.005 semi-implicit gravity.
        valid = (zi >= heights[pending, 0] + 0.03) & (zi <= heights[pending, 1] - 0.02)
        ids = pending[valid]
        start[ids], velocity[ids], target[ids] = p[valid], v[valid], end[valid]
        pending = pending[~valid]
    raise RuntimeError(f"No feasible region shots for {len(pending)} environments")
