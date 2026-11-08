import mujoco

from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import (
    booster_k1_flat_env_cfg,
)


BALL_RADIUS = 0.11

FOOTBALL_XML = """
<mujoco model="football">
  <asset>
    <texture name="ball_texture" type="2d" builtin="checker"
             rgb1="0.95 0.95 0.95" rgb2="0.08 0.08 0.08"
             width="128" height="128"/>
    <material name="ball_material" texture="ball_texture"/>
  </asset>

  <worldbody>
    <body name="football">
      <freejoint name="ball_joint"/>
      <geom name="football_geom" type="sphere"
            size="0.11" mass="0.43"
            material="ball_material"
            contype="1" conaffinity="1"
            condim="6" friction="0.6 0.005 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def football_env_cfg(play=False):
    # 使用现有 K1 的身体、关节控制和地面配置。
    cfg = booster_k1_flat_env_cfg(play=play)

    # 在机器人前方放一个具有自由运动能力的球。
    cfg.scene.entities["ball"] = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(FOOTBALL_XML),
        init_state=EntityCfg.InitialStateCfg(
            pos=(1.5, 0.0, BALL_RADIUS + 0.005),
        ),
    )
    return cfg


if __name__ == "__main__":
    cfg = football_env_cfg(play=True)
    cfg.scene.num_envs = 1

    # 只构建和检查场景，暂时不运行训练。
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()

    ball_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ball/football_geom",
    )
    assert ball_id >= 0, "Football was not added to the scene."
    assert "robot" in cfg.scene.entities, "K1 robot is missing."

    print("K1 + football scene compiled OK")
    print("Scene entities:", list(cfg.scene.entities))
    print("Robot actuators:", model.nu)
    print("Ball radius:", model.geom_size[ball_id, 0])