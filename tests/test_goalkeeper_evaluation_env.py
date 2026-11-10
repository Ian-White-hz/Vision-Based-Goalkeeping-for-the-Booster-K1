"""Real K1: balanced fixed suites, padded batches, independent state and repeatability."""

from dataclasses import asdict

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
import torch

from booster_mjlab.amp.runners.categorical_runner_cfg import PeriodicEvaluationCfg
from booster_mjlab.amp.runners.periodic_evaluation import preserve_rng
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.evaluation import GoalkeeperEvaluator


class ZeroActor(torch.nn.Module):
    def forward(self, obs, stochastic_output=False):
        assert not stochastic_output
        return torch.zeros(len(obs["actor"]), 22)


torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    obs, _ = env.reset(seed=41)
    robot, ball = env.scene["robot"], env.scene["ball"]
    before = {"joints": robot.data.joint_pos.clone(), "root": robot.data.root_link_pos_w.clone(),
              "ball": ball.data.root_link_pos_w.clone(), "actor": obs["actor"].clone(),
              "shots": env._full_goal_shots["target_position_e"].clone()}
    cfg = asdict(PeriodicEvaluationCfg(validation_seeds=(101,), holdout_seeds=(10001,),
                                     num_envs=4, max_seconds=1.5))
    rng = torch.get_rng_state().clone()
    with preserve_rng(), torch.inference_mode():
        evaluator = GoalkeeperEvaluator(training_env=env, config=cfg, device="cpu")
        first = evaluator.evaluate(ZeroActor().eval())
        second = evaluator.evaluate(ZeroActor().eval())
    assert first == second
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert env.num_envs == 2 and env.cfg.auto_reset and env.common_step_counter == 0
    for key, value in {"joints": robot.data.joint_pos, "root": robot.data.root_link_pos_w,
                       "ball": ball.data.root_link_pos_w, "actor": env.get_observations()["actor"],
                       "shots": env._full_goal_shots["target_position_e"]}.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for name in ("validation", "holdout"):
        cases = first[name]["cases"]
        assert len(cases) == 6 and sorted(r["region"] for r in cases) == list(range(6))
        assert all(r["safe_save"] == (r["save"] and not r["fall"]) for r in cases)
        assert all(not (r["save"] and r["goal"]) for r in cases)
        assert all(0 < r["duration_s"] <= 1.5 for r in cases)
    bad = dict(cfg, holdout_seeds=(101,))
    try:
        GoalkeeperEvaluator(training_env=env, config=bad, device="cpu")
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("overlapping seeds accepted")
    changed = GoalkeeperEvaluator(training_env=env, config=dict(cfg, max_seconds=2.), device="cpu")
    assert changed.signature != evaluator.signature
    print("Real K1 fixed-suite repeatability, 4-lane padded batches, six categories, physical outcomes and training state/RNG isolation passed")
finally:
    env.close()
