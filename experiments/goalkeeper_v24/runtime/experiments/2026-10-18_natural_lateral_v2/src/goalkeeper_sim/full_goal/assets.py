"""Project-owned ball and visual goal geometry; no third-party meshes."""

import mujoco
from mjlab.entity import EntityCfg

from .settings import FullGoalSettings


def ball_spec(settings: FullGoalSettings):
    spec = mujoco.MjSpec.from_string(f'''<mujoco model="full_goal_ball">
      <asset><texture name="ball_texture" type="2d" builtin="checker"
        rgb1=".95 .95 .95" rgb2=".08 .08 .08" width="128" height="128"/>
        <material name="ball_material" texture="ball_texture"/></asset>
      <worldbody><body name="football"><freejoint name="ball_joint"/>
        <geom name="football_geom" type="sphere" size="{settings.ball_radius}"
          mass="{settings.ball_mass}" material="ball_material"
          contype="1" conaffinity="1" condim="6"/>
      </body></worldbody></mujoco>''')
    spec.geom("football_geom").friction = settings.ball_friction
    return spec


def goal_spec(settings: FullGoalSettings):
    x, w, h, r = settings.goal_x, settings.goal_width / 2, settings.goal_height, settings.post_radius
    return mujoco.MjSpec.from_string(f'''<mujoco model="visual_goal"><worldbody>
      <body name="goal" mocap="true">
        <geom name="goal_line" type="box" pos="{x} 0 .003" size=".018 {w} .003"
          rgba="1 .12 .04 1" contype="0" conaffinity="0"/>
        <geom name="left_post" type="capsule" fromto="{x} {w} 0 {x} {w} {h}"
          size="{r}" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/>
        <geom name="right_post" type="capsule" fromto="{x} {-w} 0 {x} {-w} {h}"
          size="{r}" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/>
        <geom name="crossbar" type="capsule" fromto="{x} {-w} {h} {x} {w} {h}"
          size="{r}" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')


def get_ball_cfg(settings: FullGoalSettings):
    return EntityCfg(spec_fn=lambda: ball_spec(settings),
                     init_state=EntityCfg.InitialStateCfg(pos=(2.0, 0, settings.ball_radius + 0.002)))


def get_goal_cfg(settings: FullGoalSettings):
    return EntityCfg(spec_fn=lambda: goal_spec(settings))
