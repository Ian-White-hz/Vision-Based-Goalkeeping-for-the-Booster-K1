"""Real CPU PPO/AMP update receives both signs of the new movement reward."""

from dataclasses import asdict

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.amp.runners.categorical_on_policy_runner import CategoricalAmpOnPolicyRunner
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import booster_k1_goalkeeper_amp_runner_cfg
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


torch.set_num_threads(1)
torch.manual_seed(42)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(
    num_envs=2, settings=FullGoalSettings(fixed_motion_id=4),
), device="cpu")
try:
    wrapped = RslRlVecEnvWrapper(env)
    cfg = booster_k1_goalkeeper_amp_runner_cfg()
    cfg.num_steps_per_env = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    # Keep first-step limb noise small enough to preserve this controlled
    # root-velocity probe; the PPO distribution and update remain active.
    cfg.actor.distribution_cfg["init_std"] = .01
    runner = CategoricalAmpOnPolicyRunner(wrapped, asdict(cfg), log_dir=None, device="cpu")
    output_layer = [m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()
    # Put the same positive lateral velocity on opposite sides of real shot
    # targets. These are controlled start probes, not performance evaluation.
    robot = env.scene["robot"]
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    root[:, 1] += torch.tensor([-2., 2.])
    # Brief initial clearance keeps ground friction from erasing the imposed
    # velocity before the first reward sample; dynamics remain fully enabled.
    root[:, 2] += .05
    root[:, 8] = .3  # Track the capped command, rather than overspeed it.
    robot.write_root_state_to_sim(root)
    env.sim.forward()
    env.scene.update(dt=0.)
    reach = env.reward_manager.get_term_cfg("eereach")
    body_term = env.reward_manager.get_term_cfg("body_intercept")
    original = reach.func
    observed = []

    def capture(current_env, **params):
        movement = body_term.func(current_env, **body_term.params)
        raw = original(current_env, **params)
        observed.append((movement.clone(), raw.clone()))
        return raw

    reach.func = capture
    before = [{k: v.clone() for k, v in module.state_dict().items()}
              for module in (runner.alg.actor, runner.alg.critic, runner.amp_model)]
    runner.learn(1)
    assert runner.current_learning_iteration == 1 and runner.total_timesteps == 8
    assert len(observed) == 4
    assert observed[0][0][0] > 0 and observed[0][0][1] < 0, observed[0]
    assert (observed[0][1] >= 0).all(), observed[0]  # hand coverage has no signed speed bonus
    assert runner.amp_updater.expert_sampler is not None
    assert all(torch.isfinite(t).all() for row in observed for t in row)
    assert all(torch.isfinite(torch.tensor(v)) for v in runner._last_iteration_summary["ppo"].values())
    for module, state in zip((runner.alg.actor, runner.alg.critic, runner.amp_model), before):
        assert all(torch.isfinite(p).all() for p in module.parameters())
        assert any(not torch.equal(value, state[key]) for key, value in module.state_dict().items())
    print("8 real transitions: signed root progress, timed hand coverage, real experts, finite Actor/Critic/AMP updates")
finally:
    env.close()
