"""Diagnostic hand sensors detect their own collision shapes and exclude ground."""

import mjlab  # noqa: F401
import mujoco
import numpy as np
from mjlab.scene import Scene

from booster_mjlab.scripts.goalkeeper_hand_timing import add_hand_contact_sensors
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_env_cfg


cfg = booster_k1_goalkeeper_env_cfg()
add_hand_contact_sensors(cfg)
scene = Scene(cfg.scene, device="cpu")
model = scene.compile()
data = mujoco.MjData(model)
ball_adr = int(model.joint("ball/ball_joint").qposadr[0])
radius = float(model.geom("ball/football_geom").size[0])


def found(name):
    sensor = next(model.sensor(i) for i in range(model.nsensor)
                  if model.sensor(i).name.startswith(name + "_")
                  and model.sensor(i).name.endswith("_found"))
    return data.sensordata[int(sensor.adr[0])] > 0


def place(position):
    data.qpos[ball_adr:ball_adr + 3] = position
    data.qpos[ball_adr + 3:ball_adr + 7] = [1., 0., 0., 0.]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


place([3., 0., radius - .002])
assert not found("left_hand_ball_contact") and not found("right_hand_ball_contact")
assert not found("robot_ball_contact")
for side, other in (("left", "right"), ("right", "left")):
    geom = model.geom(f"robot/{side}_hand_end_collision")
    position = data.geom_xpos[geom.id].copy()
    position += np.array([radius + float(geom.size[0]) - .003, 0., 0.])
    place(position)
    assert found(f"{side}_hand_ball_contact"), side
    assert not found(f"{other}_hand_ball_contact"), other
    assert found("robot_ball_contact")
print("Real compiled sensors: left/right hand attribution and terrain exclusion passed")
