"""Source ballistic shots, fixed region assignment, reset and curriculum."""

import random
import torch
from .state import state


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
    if not s.settings.play and env.common_step_counter - s.last_step_counter > 500:
        s.startstep = 50 - random.randint(3, 10)
        s.curriculumupdate = int(
            env.episode_length_buf[ids].float().mean().item() / 50.0
        )
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

    def rand():
        return torch.rand(len(ids), device=env.device)

    scale = 1.0  # Source metres, deliberately not resized implicitly.
    start = (
        torch.stack((3 + 2 * rand(), -1.8 + 3.6 * rand(), 0.1 + 1.7 * rand()), -1)
        * scale
    )
    bounds = s.command_ranges[ids]
    end = torch.stack(
        (
            (-0.1 - 0.5 * rand()) * scale,
            bounds[:, 0] + rand() * (bounds[:, 1] - bounds[:, 0]),
            bounds[:, 2] + rand() * (bounds[:, 3] - bounds[:, 2]),
        ),
        -1,
    )
    flight = (
        (0.5 + 0.5 * torch.rand(1, device=env.device))
        if s.settings.play
        else (0.4 + 0.6 * torch.rand(1, device=env.device))
    )
    velocity = (end - start) / flight
    velocity[:, 2] += 0.5 * 9.81 * flight
    fraction = (0.1 * scale - start[:, 0:1]) / (end[:, 0:1] - start[:, 0:1])
    s.end_target[ids] = start + (end - start) * fraction + env.scene.env_origins[ids]
    ball = torch.zeros((len(ids), 13), device=env.device)
    ball[:, :3] = start + env.scene.env_origins[ids]
    ball[:, 3] = 1.0
    ball[:, 7:10] = velocity
    env.scene["ball"].write_root_state_to_sim(ball, env_ids=ids)
    s.ball_vel[ids] = velocity[:, 0]
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
        force = -0.5 * 1.225 * 0.47 * torch.pi * 0.1**2 * v.norm(dim=-1, keepdim=True) * v
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
