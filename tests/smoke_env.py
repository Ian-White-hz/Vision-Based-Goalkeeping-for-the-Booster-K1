"""Reset and step the migrated K1-plus-ball task on CPU."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import goalkeeper_sim
import goalkeeper_sim.task  # noqa: F401
from goalkeeper_sim.mdp import set_ball_state
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
    cfg = load_env_cfg(goalkeeper_sim.TASK_ID)
    cfg.scene.num_envs = 2
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observations, _ = env.reset()
        assert observations["actor"].shape == (2, 81)
        assert observations["critic"].shape == (2, 96)

        set_ball_state(
            env,
            position=(2.0, 0.0, 0.09),
            linear_velocity=(-0.4, 0.0, 0.0),
        )
        observations, reward, terminated, truncated, _ = env.step(
            torch.zeros((2, 22), device=env.device)
        )
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(reward).all()
        assert not torch.any(terminated & truncated)
        print("PASS: 2 envs, actor=81, critic=96, actions=22, exact ball state")
    finally:
        env.close()


if __name__ == "__main__":
    main()
