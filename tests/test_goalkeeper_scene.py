import mujoco

from mjlab.scene import Scene

from booster_mjlab.robots import get_k1_robot_cfg
from booster_mjlab.tasks.goalkeeper.scene.scene_cfg import (
    make_goalkeeper_scene_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)


settings = FullGoalSettings()

# 暂时沿用队友的串联 K1 场景基线。
scene_cfg = make_goalkeeper_scene_cfg(
    robot_cfg=get_k1_robot_cfg(),
    settings=settings,
    num_envs=1,
)

assert scene_cfg.num_envs == 1
assert set(scene_cfg.entities) == {"robot", "ball", "goal"}
assert scene_cfg.terrain is not None
assert scene_cfg.terrain.terrain_type == "plane"

scene = Scene(scene_cfg, device="cpu")
model = scene.compile()

# 串联 K1：22 个关节＋浮动基座；足球另有一个自由关节。
assert model.nu == 22
assert model.nq == 36
assert model.nv == 34
assert model.nmocap == 1

# Scene 给实体内部名称添加前缀，避免不同实体名称冲突。
for body_name in ("robot/Trunk", "ball/football", "goal/goal"):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    assert body_id >= 0, f"Missing body: {body_name}"

print("场景实体:", list(scene.entities))
print("nq / nv / nu:", model.nq, model.nv, model.nu)
print("K1、平地、足球和球门组合检查通过")