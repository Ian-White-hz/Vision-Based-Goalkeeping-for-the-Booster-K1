"""Randomized MuJoCo startup/reset, original ballistic bounds, delay and PPO."""

from dataclasses import asdict
import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg, runner_cfg
from booster_mjlab.tasks.goalkeeper.source.runner import SourceRunner
from booster_mjlab.tasks.goalkeeper.source.state import state

torch.set_num_threads(1)
cfg = env_cfg(6)
cfg.seed = 42
e = ManagerBasedRlEnv(cfg, device="cpu")
try:
    wrapped = RslRlVecEnvWrapper(e)
    s = state(e)
    assert torch.isfinite(e.sim.model.body_mass).all()
    # G1 source launch is ballistic for ALL six categories, including low balls.
    assert (e.scene["ball"].data.root_link_lin_vel_w[:, 0] < 0).all()
    assert (s.catchstep == 50).all()
    assert (s.kp_factor >= 0.8).all() and (s.kp_factor <= 1.2).all()
    geom_ids = e.scene["robot"].find_geoms(".*_collision")[0]
    global_ids = e.scene["robot"].indexing.geom_ids[geom_ids]
    solref = e.sim.model.geom_solref[:, global_ids]
    torch.testing.assert_close(
        solref[..., 0] * solref[..., 1], torch.full_like(solref[..., 0], 0.02)
    )
    original_regions = s.end_regions.clone()
    e.episode_length_buf[:] = 150
    e.common_step_counter = 501
    old_bounds = s.command_ranges.clone()
    wrapped.reset()
    assert s.curriculumupdate == 3
    assert (s.command_ranges[:, 0] <= old_bounds[:, 0]).all()
    torch.testing.assert_close(s.end_regions, original_regions)
    rcfg = runner_cfg()
    rcfg.num_steps_per_env = 4
    rcfg.algorithm.num_learning_epochs = 1
    rcfg.algorithm.num_mini_batches = 1
    runner = SourceRunner(wrapped, asdict(rcfg), device="cpu")
    runner.learn(1)
    action = e.action_manager.get_term("joint_pos")
    assert tuple(action.target_names) == tuple(s.robot.joint_names[i] for i in s.jids)
    assert action.substep == 4
    assert torch.isfinite(e.scene["robot"].data.joint_pos).all()
    print(
        "Randomized source simulation, curriculum, fixed regions, PD delay and HIMPPO passed"
    )
finally:
    e.close()
