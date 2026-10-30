import math

import mujoco

from booster_mjlab.tasks.goalkeeper.scene.assets import (
    get_ball_cfg,
    get_goal_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)


settings = FullGoalSettings()

assert math.isclose(settings.target_half_width, 1.07)
assert settings.episode_length_s == 16

ball_cfg = get_ball_cfg(settings)
goal_cfg = get_goal_cfg(settings)

assert math.isclose(
    ball_cfg.init_state.pos[2],
    settings.ball_radius + 0.002,
)

ball_model = ball_cfg.spec_fn().compile()
goal_model = goal_cfg.spec_fn().compile()

# 足球有一个自由关节，没有执行器。
assert ball_model.nq == 7
assert ball_model.nv == 6
assert ball_model.nu == 0

body_id = mujoco.mj_name2id(
    ball_model, mujoco.mjtObj.mjOBJ_BODY, "football"
)
geom_id = mujoco.mj_name2id(
    ball_model, mujoco.mjtObj.mjOBJ_GEOM, "football_geom"
)

assert body_id >= 0
assert geom_id >= 0
assert math.isclose(
    float(ball_model.body_mass[body_id]),
    settings.ball_mass,
)
assert math.isclose(
    float(ball_model.geom_size[geom_id, 0]),
    settings.ball_radius,
)
assert ball_model.geom_contype[geom_id] == 1
assert ball_model.geom_conaffinity[geom_id] == 1

# 球门可定位，但不参与碰撞。
assert goal_model.nmocap == 1
assert (goal_model.geom_contype == 0).all()
assert (goal_model.geom_conaffinity == 0).all()

print("足球质量:", ball_model.body_mass[body_id])
print("足球半径:", ball_model.geom_size[geom_id, 0])
print("默认回合时长:", settings.episode_length_s)
print("场景参数与资产构造检查通过")