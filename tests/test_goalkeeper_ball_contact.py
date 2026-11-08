"""Compile the real sensor and verify it sees robot contacts, not the ground."""

import mjlab  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)


cfg = booster_k1_goalkeeper_env_cfg()
sensor_cfg = next(s for s in cfg.scene.sensors if s.name == "robot_ball_contact")
assert sensor_cfg.history_length == cfg.decimation
assert sensor_cfg.secondary.entity == "robot"
assert sensor_cfg.secondary.mode == "subtree"
assert sensor_cfg.secondary_policy == "error"

scene = Scene(cfg.scene, device="cpu")
model = scene.compile()
data = mujoco.MjData(model)
ball_adr = int(model.joint("ball/ball_joint").qposadr[0])
found_sensor = next(
    model.sensor(i)
    for i in range(model.nsensor)
    if model.sensor(i).name.startswith("robot_ball_contact_")
    and model.sensor(i).name.endswith("_found")
)
sensor_adr = int(found_sensor.adr[0])
ball_radius = float(model.geom("ball/football_geom").size[0])


def place_ball(position):
    data.qpos[ball_adr : ball_adr + 3] = position
    data.qpos[ball_adr + 3 : ball_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# A genuine ball-terrain contact exists, but the robot-filtered sensor is zero.
place_ball([3.0, 0.0, ball_radius - 0.002])
ball_body_id = int(model.body("ball/football").id)
assert any(
    ball_body_id in model.geom_bodyid[data.contact[i].geom] for i in range(data.ncon)
)
assert data.sensordata[sensor_adr] == 0.0

# Place the ball against the real K1 trunk collision box.
trunk_geom = model.geom("robot/trunk_collision")
center = data.geom_xpos[trunk_geom.id].copy()
rotation = data.geom_xmat[trunk_geom.id].reshape(3, 3)
offset = np.array([float(trunk_geom.size[0]) + ball_radius - 0.005, 0.0, 0.0])
place_ball(center + rotation @ offset)
assert data.sensordata[sensor_adr] > 0.0

print("Real K1 robot-ball contact sensor: robot detected, terrain excluded")

# Exercise the actual batched sensor/history API without using the independently
# changing mixed six-region distribution. This rolling mode is already supported.
env_cfg = booster_k1_goalkeeper_env_cfg(num_envs=2, shot_mode="rolling")
env = ManagerBasedRlEnv(env_cfg, device="cpu")
try:
    env.reset(seed=42)
    for _ in range(3):
        _, reward, _, _, _ = env.step(torch.zeros(2, 22))
        contact = env.scene["robot_ball_contact"].data
        assert contact.found.shape == (2, 1)
        assert contact.force_history.shape == (2, 1, env_cfg.decimation, 3)
        assert torch.isfinite(reward).all()
        # The rolling balls are still far from the robot; terrain contacts must
        # not produce save flags via the shared reward/metrics path.
        assert not (contact.found > 0).any()
        assert not env._goalkeeper_shot_state.success_flag.any()

    # A slow incoming ball against the real trunk must now earn a save through
    # the actual physics -> contact history -> shot state -> reward/metrics path.
    ball_state = torch.zeros(1, 13)
    ball_state[0, :3] = env.scene["robot"].data.root_link_pos_w[0] + torch.tensor(
        [0.165, 0.0, 0.1]
    )
    ball_state[0, 3] = 1.0
    ball_state[0, 7] = -0.4
    env.scene["ball"].write_root_state_to_sim(ball_state, env_ids=torch.tensor([0]))
    contact_seen = False
    save_reward = 0.0
    for _ in range(3):
        env.step(torch.zeros(2, 22))
        contact = env.scene["robot_ball_contact"].data
        contact_seen |= bool((contact.force_history[0].abs() > 1.0e-6).any())
        save_reward += env._goalkeeper_shot_state.stopball_reward[0].item()
    assert contact_seen
    assert env._goalkeeper_shot_state.success_flag.tolist() == [True, False]
    assert save_reward == 1.0

    env.reset(env_ids=torch.tensor([0]))
    assert not env.scene["robot_ball_contact"].data.force_history[0].any()
    assert not env._goalkeeper_shot_state.success_flag.any()
finally:
    env.close()

print("Batched physics/history, real rolling save, one-shot reward and reset passed")
