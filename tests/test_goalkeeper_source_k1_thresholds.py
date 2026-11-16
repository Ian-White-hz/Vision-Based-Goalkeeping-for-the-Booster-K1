"""K1 ball-phase thresholds: far phase, speed-relative deflection, behind line."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg, runner_cfg
from booster_mjlab.tasks.goalkeeper.source.state import state
from booster_mjlab.tasks.goalkeeper.source.settings import (
    PHASE1_X,
    DEFLECT_RATIO,
    DEFLECT_MIN,
    BEHIND_X,
    BALL_VISIBLE_X,
    SHOT_START_X,
    REGIONS,
)
from booster_mjlab.tasks.goalkeeper.source.events import sample_region_shots

torch.set_num_threads(1)

# Reduced exploration noise for the small-batch K1 setup.
_rcfg = runner_cfg()
assert _rcfg.actor.distribution_cfg["init_std"] == 0.5
assert _rcfg.algorithm.entropy_coef == 0.003

# Shot shape: launches are visible to the actor and arrive as flat-ish shots.
assert BALL_VISIBLE_X > SHOT_START_X[1] + 0.2
torch.manual_seed(0)
ids = torch.arange(600) % 6
p, v, _ = sample_region_shots(torch.tensor(REGIONS)[ids], ids)
ti = -p[:, 0] / v[:, 0]
vz = v[:, 2] - 9.81 * ti
descent = torch.rad2deg(torch.atan2(-vz, -v[:, 0]))
assert descent.median() < 40 and torch.quantile(descent, 0.9) < 48
assert (-v[:, 0]).median() > 4.5
cfg = env_cfg(num_envs=6, randomize=False)
cfg.seed = 7
env = ManagerBasedRlEnv(cfg, device="cpu")
try:
    wrapped = RslRlVecEnvWrapper(env)
    wrapped.step(torch.zeros(6, 22))
    s = state(env)
    s.refresh()
    ox = s.env_origins[:, 0]

    def place(x, vx, vx0, arm=True):
        s.ball_states[:, 0] = ox + x
        s.ball_states[:, 7] = vx
        s.ball_vel[:] = vx0
        s.stop_flag.zero_()
        s.success_flag.zero_()
        s.arm_touch[:] = arm  # threshold checks assume an arm touch

    # Slow K1 shot: a 1.2 m/s loss (44%) is a deflection; upstream needed >2.
    place(0.05, -1.5, -2.7)
    assert s._deflected().all() and not s._behind().logical_not().any()
    assert (s._reward_stopball() == 1).all() and (s.success_flag == 1).all()
    assert (s._reward_stopball() == 0).all()  # one-shot
    # Same deflection without an arm touch: credited (upstream, any body part).
    place(0.05, -1.5, -2.7, arm=False)
    assert (s._reward_stopball() == 1).all() and (s.success_flag == 1).all()
    # Arm-only mode: the window closes without credit.
    s.arm_save_credit = True
    place(0.05, -1.5, -2.7, arm=False)
    assert (s._reward_stopball() == 0).all() and (s.success_flag == 0).all()
    assert (s.stop_flag == 1).all()
    s.arm_save_credit = False
    # Small loss (20%) is not.
    place(0.05, -2.16, -2.7)
    assert not s._deflected().any() and (s._reward_stopball() == 0).all()
    # Floor for very slow balls.
    place(0.05, -0.6, -1.0)
    assert not s._deflected().any()
    torch.testing.assert_close(
        torch.tensor(DEFLECT_RATIO * 1.0).clamp(min=DEFLECT_MIN), torch.tensor(0.5)
    )
    # Deflection just behind the robot plane but before the goal line counts.
    place(BEHIND_X + 0.02, -1.0, -2.7)
    assert (s._reward_stopball() == 1).all()
    # Ball past the goal line: stopped, no reward.
    place(BEHIND_X - 0.05, -1.0, -2.7)
    assert (s._reward_stopball() == 0).all() and (s.stop_flag == 1).all()
    assert (s.success_flag == 0).all() and s._behind().all()

    # Far phase active above PHASE1_X, reach phase below it.
    s.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    s.dist[:] = 2.0  # sigmoid reach ~0
    s.end_target[:] = s.torso_pos
    place(PHASE1_X + 0.1, -2.7, -2.7)
    far = s._reward_eereach()
    place(PHASE1_X - 0.1, -2.7, -2.7)
    near = s._reward_eereach()
    assert (far > 0.5).all() and (near < 0.01).all()

    # Curriculum EMA: first finished batch initialises, later ones are smoothed.
    s.episode_length_ema = -1.0
    env.episode_length_buf[:] = 150
    env.common_step_counter = s.last_step_counter + 501
    wrapped.reset()
    assert s.curriculumupdate == 3
    env.episode_length_buf[:] = 10
    env.common_step_counter = s.last_step_counter + 501
    wrapped.reset()
    assert s.episode_length_ema > 140 and s.curriculumupdate == 2
    print("K1 ball-phase thresholds and curriculum EMA passed")
finally:
    env.close()
