import sys

import mujoco
import mjlab.tasks
from copy import deepcopy
from dataclasses import replace

from mjlab.envs.mdp import events, terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    booster_k1_ppo_runner_cfg,
)
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from football_scene import BALL_RADIUS, football_env_cfg


TASK_NAME = "K1-Football-Stand"


def stand_env_cfg(play=False):
    # 使用训练版配置，保留原有跌倒判断。
    cfg = football_env_cfg(play=False)
    # 给地面设置固定摩擦。
    terrain = cfg.scene.terrain
    assert terrain is not None
    terrain.collisions = tuple(
        replace(collision, friction=(0.8, 0.005, 0.0001))
        for collision in terrain.collisions
    )

    # 保留原框架的脚底摩擦设置，固定为 0.8。
    foot_friction = deepcopy(cfg.events["foot_friction"])
    foot_friction.params["ranges"] = (0.8, 0.8)
    cfg.scene.num_envs = 1024
    cfg.scene.env_spacing = 6.0
    cfg.episode_length_s = 10.0
    cfg.sim.nconmax = 128

    # 站立阶段：足球放在远处，不发球。
    cfg.scene.entities["ball"].init_state.pos = (
        2.0, 0.0, BALL_RADIUS + 0.002
    )

     # 每回合重置；启动时设置固定脚底摩擦。
    cfg.events = {
        "reset_scene": EventTermCfg(
            func=events.reset_scene_to_default,
            mode="reset",
        ),
        "foot_friction": foot_friction,
    }
    cfg.curriculum = {}

    # 所有环境都要求站立。
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0
    twist.ranges.lin_vel_x = (-0.1, 0.1)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)

    # 站立阶段不需要抬脚和迈步奖励。
    for name in ("air_time", "foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(name, None)

    cfg.terminations["fallen"] = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={"limit_angle": 1.0},
    )

    if play:
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False

    return cfg


def check_scene():
    cfg = stand_env_cfg()
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()

    print("Standing scene compiled OK")
    print("Entities:", list(cfg.scene.entities))
    print("Actuators:", model.nu)
    print("Rewards:", list(cfg.rewards))
    print("Terminations:", list(cfg.terminations))

    # 检查之前警告里提到的 geom 0。
    print(
        "Geom 0:",
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 0),
        "friction:", model.geom_friction[0],
        "condim:", model.geom_condim[0],
        "contype:", model.geom_contype[0],
        "conaffinity:", model.geom_conaffinity[0],
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Choose: check, train, or play")

    command = sys.argv.pop(1)

    if command == "check":
        check_scene()
        return

    if command not in ("train", "play"):
        raise SystemExit("Choose: check, train, or play")

    agent = booster_k1_ppo_runner_cfg()
    agent.experiment_name = "k1_football_stand"
    agent.max_iterations = 2000
    agent.logger = "tensorboard"

    register_mjlab_task(
        task_id=TASK_NAME,
        env_cfg=stand_env_cfg(),
        play_env_cfg=stand_env_cfg(play=True),
        rl_cfg=agent,
        runner_cls=BoosterOnPolicyRunner,
    )

    sys.argv.insert(1, TASK_NAME)

    if command == "train":
        from booster_mjlab.scripts.train import main as entry
    else:
        from booster_mjlab.scripts.play import main as entry

    entry()


if __name__ == "__main__":
    main()