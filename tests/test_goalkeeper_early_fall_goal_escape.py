"""Counterfactual probe: early fall discards a pending goal, not a same-step goal.

Injected terminal poses isolate reward bookkeeping, not a learned policy's intent.
All production rewards, events, terminations and auto-reset remain enabled.
"""

import json
import math
from pathlib import Path

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
)


torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=3), device="cpu")
try:
    env.reset(seed=42)
    robot = env.scene["robot"]
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    # Move all robots away from the goal-bound trajectory: neither branch blocks it.
    root[:, 0] = env.scene.env_origins[:, 0] + 2.0
    angle = math.radians(75) / 2
    root[[0, 2], 3:7] = torch.tensor([math.cos(angle), math.sin(angle), 0., 0.])
    root[[0, 2], 2] += .4
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root)
    ball = torch.zeros((3, 13))
    ball[:, :3] = env.scene.env_origins + torch.tensor([-.65, 0., .4])
    ball[2, 0] = env.scene.env_origins[2, 0] - 1.0
    ball[:, 3] = 1
    ball[:, 7] = -2
    env.scene["ball"].write_root_state_to_sim(ball)
    env._full_goal_shots["velocity"][:] = ball[:, 7:10]
    env.sim.forward()
    env.scene.update(dt=0.)

    action = torch.zeros((3, env.action_manager.total_action_dim))
    records = []
    for step in range(12):
        _, task, terminated, truncated, _ = env.step(action)
        terms = {
            name: (env.reward_manager._step_reward[:, i] * env.step_dt).clone()
            for i, name in enumerate(env.reward_manager.active_terms)
        }
        records.append({
            "step": step + 1, "time": (step + 1) * env.step_dt,
            "task": task.tolist(), "fall": terms["fall"].tolist(),
            "goal": terms["goal_conceded"].tolist(),
            "terminated": terminated.tolist(), "truncated": truncated.tolist(),
            "ball_x_after_possible_reset": (
                env.scene["ball"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
            ).tolist(),
        })
        if step == 0:
            assert terminated.tolist() == [True, False, True]
            assert not truncated.any()
            torch.testing.assert_close(terms["fall"], torch.tensor([-5., 0., -5.]))
            torch.testing.assert_close(terms["goal_conceded"], torch.tensor([0., 0., -2.]))
            assert not env._goalkeeper_outcome_state.goal_conceded[[0, 2]].any()
            assert (env.scene["ball"].data.root_link_pos_w[[0, 2], 0]
                    - env.scene.env_origins[[0, 2], 0] > 0).all()
        assert not terminated[1] and not truncated[1], "Control must survive the goal crossing"

    assert sum(r["goal"][1] for r in records) == -2.
    assert sum(r["goal"][0] for r in records) == 0.
    result = {
        "lanes": ["early_fall_before_goal", "upright_unblocked_ball", "same_step_fall_and_goal"],
        "first_step_event_task": [-5., 0., -7.],
        "event_task_over_probe": [sum(r["fall"][i] + r["goal"][i] for r in records) for i in range(3)],
        "records": records,
        "scope": "Injected states verify censored goal cost; not evidence of learned intent or full AMP return preference.",
    }
    output = Path("artifacts/goalkeeper_early_fall_goal_escape/goal_penalty_2_result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    print(f"PASS: {output}")
finally:
    env.close()
