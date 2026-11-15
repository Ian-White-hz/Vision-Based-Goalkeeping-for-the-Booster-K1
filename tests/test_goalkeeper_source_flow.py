"""Source reward parity, all six regions, real MuJoCo + HIMPPO + checkpoint."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import ast
import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg, runner_cfg
from booster_mjlab.tasks.goalkeeper.source.state import state
from booster_mjlab.tasks.goalkeeper.source.runner import SourceRunner
from booster_mjlab.tasks.goalkeeper.source.settings import REWARD_SCALES, AMP_JOINT_IDS

torch.set_num_threads(1)
cfg = env_cfg(num_envs=6, randomize=False)
cfg.seed = 42
env = ManagerBasedRlEnv(cfg, device="cpu")
try:
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    assert (
        obs["actor"].shape == (6, 750)
        and obs["critic"].shape == (6, 92)
        and obs["amp"].shape == (6, len(AMP_JOINT_IDS))
    )
    s = state(env)
    assert s.end_regions.tolist() == list(range(6))
    assert torch.allclose(obs["critic"][:, 78] * 3, s.end_regions.float())
    assert env.step_dt == 0.02 and env.cfg.episode_length_s == 3
    assert set(env.cfg.rewards) == set(REWARD_SCALES)
    assert not {
        "goal_conceded",
        "fall_termination",
        "body_intercept",
        "body_tilt",
    } & set(env.cfg.rewards)
    # Existing source checkout is an independent oracle; no IsaacGym import.
    original = Path(
        "../Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py"
    )
    if original.exists():
        from mjlab.utils.lab_api.math import quat_apply_inverse

        tree = ast.parse(original.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_reward_" + k for k in REWARD_SCALES}
        ]
        ns = {"torch": torch, "quat_rotate_inverse": quat_apply_inverse}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                str(original),
                "exec",
            ),
            ns,
        )
        wrapped.step(torch.zeros(6, 22))
        s.refresh()
        # K1 rescales ball-phase thresholds; upstream values must reproduce it.
        upstream = dict(phase1_x=1.5, deflect_ratio=0.0, deflect_min=2.0, behind_x=0.0)
        for k, v in upstream.items():
            setattr(s, k, v)
        for name in REWARD_SCALES:
            if name == "successland":
                continue  # K1 adaptation covered by test_goalkeeper_source_small_jump.py.
            # Snapshot source stateful flags for identical preconditions.
            flags = [s.stop_flag.clone(), s.success_flag.clone(), s.has_in_air.clone()]
            expected = ns["_reward_" + name](s)
            s.stop_flag.copy_(flags[0])
            s.success_flag.copy_(flags[1])
            s.has_in_air.copy_(flags[2])
            actual = getattr(s, "_reward_" + name)()
            torch.testing.assert_close(actual, expected)
        for k in upstream:
            delattr(s, k)
        print("All applicable source reward methods match upstream on real K1 state")
    else:
        print("SKIP upstream checkout parity (standalone smoke remains active)")
    # Monotonic jump velocity bonus, source low-region hand target (not feet).
    s.ball_states[:, 0] = s.env_origins[:, 0] + 0.5  # reach phase (below PHASE1_X)
    s.ball_vel[:] = -5.0
    s.ball_states[:, 7] = -5.0
    s.dist[:] = 0.2
    s.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    s.rigid_body_states[:, s.upper_body_index, 7:10] = 0.0
    standing = s._reward_eereach().clone()
    s.rigid_body_states[2:4, s.upper_body_index, 9] = 0.5
    jumping = s._reward_eereach()
    assert (jumping[2:4] > standing[2:4]).all()
    wrapped.reset()
    rcfg = runner_cfg()
    rcfg.num_steps_per_env = 4
    rcfg.algorithm.num_learning_epochs = 1
    rcfg.algorithm.num_mini_batches = 1
    with TemporaryDirectory() as directory:
        runner = SourceRunner(wrapped, asdict(rcfg), directory, device="cpu")
        before = {k: v.clone() for k, v in runner.alg.actor_critic.state_dict().items()}
        runner.learn(1)
        assert runner.current_learning_iteration == 1
        for prefix in (
            "actor.",
            "critic.",
            "ball_estimator.",
            "region_estimator.",
            "history_encoder.",
        ):
            assert any(
                not torch.equal(v, runner.alg.actor_critic.state_dict()[k])
                for k, v in before.items()
                if k.startswith(prefix)
            ), prefix
        ckpt = Path(directory) / "model_1.pt"
        restored = SourceRunner(wrapped, asdict(rcfg), device="cpu")
        restored.load(ckpt)
        assert restored.current_learning_iteration == 1
        torch.testing.assert_close(
            runner.get_inference_policy()(wrapped.get_observations()),
            restored.get_inference_policy()(wrapped.get_observations()),
        )
        restored.learn(1)
        assert restored.current_learning_iteration == 2
    print(
        "Source K1: six regions, 750/92/22 obs, estimator/classifier gradients, real PPO/AMP, save/resume passed"
    )
finally:
    env.close()
