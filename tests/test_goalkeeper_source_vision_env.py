"""Vision ball input in the source env: truth unchanged, synthetic and rendered modes."""

from dataclasses import asdict

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.tasks.goalkeeper.source.config import env_cfg, runner_cfg
from booster_mjlab.tasks.goalkeeper.source.runner import SourceRunner
from booster_mjlab.tasks.goalkeeper.source.state import state
from booster_mjlab.tasks.goalkeeper.source.vision import camera

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def run(mode, n, steps, check):
    cfg = env_cfg(num_envs=n, play=True, randomize=False, ball_input=mode)
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg, device=DEV)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner = SourceRunner(wrapped, asdict(runner_cfg()), device=DEV)
        policy = runner.alg.actor_critic
        obs = wrapped.get_observations()
        s = state(env)
        out = []
        with torch.no_grad():
            for _ in range(steps):
                obs, _, _, _ = wrapped.step(policy.act_inference(obs["actor"]))
                assert obs["actor"].shape == (n, 750) and obs["critic"].shape == (n, 92)
                out.append(check(env, s, obs))
        return out
    finally:
        env.close()


# Truth: no ball input object, actor ball equals the critic's (noise off in play).
run("truth", 4, 3, lambda env, s, obs: None)
cfg = env_cfg(num_envs=4, play=True, randomize=False)
assert cfg.source_settings.ball_input == "truth"
assert not any(getattr(x, "name", "") == camera.CAMERA_NAME for x in cfg.scene.sensors)


def synthetic_check(env, s, obs):
    bi = s.ball_input
    ball = obs["actor"][:, -75:-72]
    seen = bi.seen
    assert torch.equal(ball[~seen], torch.zeros_like(ball[~seen]))
    err = (ball - bi.true_local).norm(dim=-1)
    return seen.clone(), err, bi.true_local[:, 0].clone()


rows = run("synthetic", 64, 45, synthetic_check)
seen = torch.stack([r[0] for r in rows])
err = torch.stack([r[1] for r in rows])
x = torch.stack([r[2] for r in rows])
near = seen & (x < 2.0) & (x > 0.3)
assert seen[10:35].float().mean() > 0.9, seen.float().mean(1)
assert err[near].median() < 0.15, err[near].median()
print(
    f"synthetic: seen {seen.float().mean():.2f}, error median near "
    f"{err[near].median():.3f} m, far {err[seen & (x > 3)].median():.3f} m"
)


def rendered_check(env, s, obs):
    bi = s.ball_input
    cid = bi.sensor.camera_idx
    pos, rot = bi.camera_pose()
    torch.testing.assert_close(env.sim.data.cam_xpos[:, cid], pos, atol=1e-4, rtol=0)
    rot_mj = env.sim.data.cam_xmat[:, cid].reshape(-1, 3, 3)
    torch.testing.assert_close(camera.mujoco_cam_to_cv(rot_mj), rot, atol=1e-4, rtol=0)
    return (
        bi.seen.clone(),
        (obs["actor"][:, -75:-72] - bi.true_local).norm(dim=-1),
        bi.true_local[:, 0].clone(),
    )


rows = run("rendered", 8, 40, rendered_check)
seen = torch.stack([r[0] for r in rows])
err = torch.stack([r[1] for r in rows])
x = torch.stack([r[2] for r in rows])
near = seen & (x < 2.0) & (x > 0.3)
assert seen[10:35].float().mean() > 0.8 and err[near].median() < 0.15
print(
    f"rendered (segmentation detector): seen {seen.float().mean():.2f}, error median near {err[near].median():.3f} m"
)
print("vision env passed")
