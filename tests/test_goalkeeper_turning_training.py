"""One real CPU PPO+AMP update with the new heading cost active."""

from dataclasses import asdict

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import (
    booster_k1_goalkeeper_amp_runner_cfg,
)


torch.set_num_threads(1)
torch.manual_seed(42)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    wrapped = RslRlVecEnvWrapper(env)
    cfg = booster_k1_goalkeeper_amp_runner_cfg()
    cfg.num_steps_per_env = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    runner = CategoricalAmpOnPolicyRunner(
        wrapped, asdict(cfg), log_dir=None, device="cpu"
    )
    robot = env.scene["robot"]
    state = torch.cat(
        (robot.data.root_link_pose_w, robot.data.root_link_vel_w), dim=-1
    ).clone()
    angle = torch.deg2rad(torch.tensor([55.0, -55.0])) / 2
    state[:, 3:7] = torch.stack(
        (angle.cos(), angle * 0, angle * 0, angle.sin()), dim=-1
    )
    robot.write_root_state_to_sim(state)
    env.sim.forward()
    env.scene.update(dt=0.0)
    runner.learn(1)
    assert runner.current_learning_iteration == 1
    assert runner.total_timesteps == 8
    assert runner.amp_updater.expert_sampler is not None
    heading_index = env.reward_manager.active_terms.index("heading_deviation")
    assert (env.reward_manager._step_reward[:, heading_index] < 0).all()
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in runner._last_iteration_summary["ppo"].values()
    )
    for module in (runner.alg.actor, runner.alg.critic, runner.amp_model):
        assert all(torch.isfinite(p).all() for p in module.parameters())
    print(
        "真实环境 8 条 transition：朝向代价生效，PPO/AMP 联合更新及真实 expert 接口通过"
    )
finally:
    env.close()
