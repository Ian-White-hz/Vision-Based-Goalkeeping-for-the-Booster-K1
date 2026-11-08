"""Reuse the saved ball/goal scene for independent random rolling shots.

Only ordinary episode resets initialize a shot. The sampled endpoint and reset
metadata are for physical initialization/evaluation, never the RGB controller.
"""
import torch
from mjlab.managers import EventTermCfg
from coordinated_capture_task import env_class, make_cfg as narrow_cfg, measurements, raw_class
from shot_curriculum import RANGES, sample_shots, describe, specification
from goalkeeper_sim.full_goal.settings import FullGoalSettings


def launch_random(env, env_ids, stage='angled_near'):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not hasattr(env, '_incoming_ready'):
        env._incoming_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._incoming_touched = env._incoming_ready.clone()
        env._incoming_initial_x = torch.zeros(env.num_envs, device=env.device)
        env._incoming_initial_speed = env._incoming_initial_x.clone()
    if not hasattr(env, '_incoming_launch_start'):
        env._incoming_launch_start = torch.zeros((env.num_envs, 3), device=env.device)
        env._incoming_goal_target = torch.zeros_like(env._incoming_launch_start)
        env._incoming_guard_y = torch.zeros(env.num_envs, device=env.device)
    settings = FullGoalSettings()
    if stage not in ('goal_gate_clear_miss_eval','goal_gate_clear_miss_eval_v2','goal_width_fast_edge_mix_v2'):
        assert RANGES[stage].target_y_half_width <= settings.target_half_width+1e-7
    sides = torch.where(ids % 2 == 0, -1., 1.) if getattr(env, '_balanced_eval', False) else None
    shot = sample_shots(len(ids), stage, env.device, balanced_sides=sides,
        goal_x=settings.goal_x, ball_radius=settings.ball_radius)
    state = torch.zeros((len(ids), 13), device=env.device)
    state[:, :3] = shot['start']+env.scene.env_origins[ids]
    state[:, 3] = 1.
    state[:, 7:9] = shot['velocity']
    state[:, 10:13] = shot['angular_velocity']
    env._bridge_left[ids] = shot['guard_y'] > 0
    env._incoming_initial_x[ids] = shot['start'][:, 0]
    env._incoming_initial_speed[ids] = shot['speed']
    env._incoming_ready[ids] = env._incoming_touched[ids] = False
    # Central shots may already be aligned. Keep measured stop and all physical
    # hold gates, without imposing a fictitious minimum lateral travel distance.
    env._incoming_allow_stationary = True
    env._incoming_launch_start[ids] = shot['start']
    env._incoming_goal_target[ids] = shot['target']
    env._incoming_guard_y[ids] = shot['guard_y']
    env.scene['ball'].write_root_state_to_sim(state, env_ids=ids)


def make_cfg(num_envs=24, stage='angled_near'):
    assert stage in RANGES
    cfg = narrow_cfg(num_envs)
    cfg.events['ball_reset'] = EventTermCfg(func=launch_random, mode='reset', params={'stage':stage})
    return cfg
