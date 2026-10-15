"""Exercise real resets, partial resets, rolling launch, stepping and goal reset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from mjlab.envs import ManagerBasedRlEnv
from goalkeeper_sim.env_cfg import goalkeeper_env_cfg
from goalkeeper_sim.full_goal.env_cfg import full_goal_env_cfg
from goalkeeper_sim.full_goal.settings import FullGoalSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    settings = FullGoalSettings()
    base = goalkeeper_env_cfg()
    original_events = set(base.events)
    cfg = full_goal_env_cfg(play=True, num_envs=2, policy_env_cfg=base, enable_camera=args.camera)
    assert set(base.events) == original_events and "goal" not in base.scene.entities
    assert cfg.actions == base.actions and cfg.commands == base.commands
    assert cfg.episode_length_s == 16
    if args.camera:
        sensor = next(s for s in cfg.scene.sensors if s.name == "keeper_rgb")
        assert (sensor.width, sensor.height, sensor.fovy) == (128, 96, 100.)
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if args.camera else "cpu")
    try:
        observations, _ = env.reset()
        ball = env.scene["ball"]
        state = torch.cat((ball.data.root_link_pose_w, ball.data.root_link_vel_w), dim=1).clone()
        sample = env._full_goal_shots
        local = ball.data.root_link_pos_w - env.scene.env_origins
        assert ((local[:, 0] >= 1.8) & (local[:, 0] <= 2.6)).all()
        assert (local[:, 1].abs() <= 1.4).all()
        assert torch.allclose(state[:, 7:9], sample["velocity"], atol=1e-5)
        assert torch.allclose(state[:, 10], -state[:, 8] / .11, atol=1e-5)
        assert torch.allclose(state[:, 11], state[:, 7] / .11, atol=1e-5)
        assert (state[:, 12] == 0).all()
        old_other = {k: v[1].clone() for k, v in sample.items()}
        from goalkeeper_sim.full_goal.events import launch_shots
        launch_shots(env, torch.tensor([0], device=env.device), settings)
        assert all(torch.equal(env._full_goal_shots[k][1], old_other[k]) for k in old_other)
        for _ in range(3):
            observations, reward, _, _, _ = env.step(torch.zeros((2, 22), device=env.device))
            assert observations["actor"].shape == (2, 81)
            assert torch.isfinite(observations["actor"]).all() and torch.isfinite(reward).all()
        # Put the whole ball behind the line, then let normal step/auto-reset run.
        state = torch.cat((ball.data.root_link_pose_w, ball.data.root_link_vel_w), dim=1).clone()
        state[:, :3] = env.scene.env_origins + state.new_tensor((-1.1, 0, .11))
        state[:, 7:13] = 0
        ball.write_root_state_to_sim(state)
        env.sim.forward()
        env.step(torch.zeros((2, 22), device=env.device))
        assert env.termination_manager.get_term("goal_conceded").all()
        assert env._full_goal_outcome.last["goal"].all()
        if args.camera:
            env.sim.sense()
            assert env.scene["keeper_rgb"].data.rgb.shape == (2, 96, 128, 3)
        print("PASS: real full-goal scene, independent origins, pure rolling, partial reset, goal auto-reset",
              "RGB" if args.camera else "CPU")
    finally:
        env.close()


if __name__ == "__main__":
    main()
