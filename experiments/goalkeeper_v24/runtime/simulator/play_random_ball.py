import math
import sys

import torch
import mjlab.tasks

from mjlab.envs.mdp import events, terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer import ViewerConfig

from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    booster_k1_ppo_runner_cfg,
)
from booster_mjlab.rl.runner import BoosterOnPolicyRunner

from football_scene import BALL_RADIUS, football_env_cfg


TASK_NAME = "K1-Football-Random-Debug"


def launch_ball(env, env_ids):
    """每轮重新设置来球位置、方向、速度和滚动转速。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = len(env_ids)
    if n == 0:
        return

    # 前一个事件已经重置机器人，刷新后再读取它的位置。
    env.sim.forward()
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]
    heading = robot.data.heading_w[env_ids]

    def uniform(low, high):
        return low + (high - low) * torch.rand(
            n, device=env.device
        )

    # 来球起点：前方左右 45 度、距离 1.2～1.8 米。
    angle = heading + uniform(-math.pi / 4, math.pi / 4)
    distance = uniform(1.2, 1.8)
    start_xy = robot_xy + distance[:, None] * torch.stack(
        (torch.cos(angle), torch.sin(angle)), dim=1
    )

    # 瞄准机器人脚附近的随机左右位置。
    side = uniform(-0.15, 0.15)
    target_xy = robot_xy + torch.stack(
        (-torch.sin(heading) * side,
         torch.cos(heading) * side),
        dim=1,
    )

    direction = target_xy - start_xy
    direction = direction / direction.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-6)

    speed = uniform(0.4, 0.8)
    velocity_xy = direction * speed[:, None]

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :2] = start_xy
    pose[:, 2] = (
        env.scene.env_origins[env_ids, 2] + BALL_RADIUS + 0.002
    )
    pose[:, 3] = 1.0

    velocity = torch.zeros(n, 6, device=env.device)
    velocity[:, :2] = velocity_xy

    # 转动方向与前进方向配合，让球从一开始就在滚动。
    velocity[:, 3] = -velocity_xy[:, 1] / BALL_RADIUS
    velocity[:, 4] = velocity_xy[:, 0] / BALL_RADIUS

    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def random_ball_cfg(play=False):
    cfg = football_env_cfg(play=play)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 6.0

    # 初步测试采用固定的机器人初始姿态。
    cfg.events = {
        "reset_scene": EventTermCfg(
            func=events.reset_scene_to_default,
            mode="reset",
        ),
        "launch_ball": EventTermCfg(
            func=launch_ball,
            mode="reset",
        ),
    }
    cfg.curriculum = {}

    # 暂时不给机器人随机的行走指令。
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0
    twist.ranges.lin_vel_x = (-0.1, 0.1)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)

    # 测试时每六秒重置，包括已经倒下的机器人。
    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=terminations.time_out,
            time_out=True,
        ),
    }

    cfg.sim.nconmax = 128

    # 固定观察整个场地，避免镜头跟着倒下的机器人移动。
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat = (0.4, 0.0, 0.5)
    cfg.viewer.distance = 3.0

    return cfg


if __name__ == "__main__":
    agent = booster_k1_ppo_runner_cfg()
    agent.experiment_name = "k1_football_random_debug"

    register_mjlab_task(
        task_id=TASK_NAME,
        env_cfg=random_ball_cfg(),
        play_env_cfg=random_ball_cfg(play=True),
        rl_cfg=agent,
        runner_cls=BoosterOnPolicyRunner,
    )

    # 只运行动态查看器，不启动训练、不加载旧模型。
    sys.argv[1:] = [
        TASK_NAME,
        "--agent", "zero",
        "--num-envs", "1",
        "--device", "cpu",
        "--viewer", "viser",
        "--log-root", "logs/football_debug",
    ]

    from booster_mjlab.scripts.play import main
    main()