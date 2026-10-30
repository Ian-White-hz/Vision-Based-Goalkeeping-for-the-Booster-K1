import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_recovery_mask,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)


settings = FullGoalSettings()
cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=1,
    settings=settings,
)

# 保留结束时的状态，避免自动重置后读到下一次发球。
cfg.auto_reset = False

env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    env.reset(seed=42)

    ball = env.scene["ball"]
    robot = env.scene["robot"]

    launch_vx = env._full_goal_shots["velocity"][0, 0].item()
    print(f"初始 vx: {launch_vx:.3f} m/s")
    print(f"Episode 时长: {settings.episode_length_s} s")

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    reached_guard_region = False
    crossed_reference = False
    recovery_seen = False
    minimum_x = float("inf")

    print_every = max(1, round(1.0 / env.step_dt))

    for step in range(env.max_episode_length):
        _, reward, terminated, truncated, _ = env.step(action)

        position_e = (
            ball.data.root_link_pos_w[0]
            - env.scene.env_origins[0]
        )
        velocity_w = ball.data.root_link_lin_vel_w[0]

        assert torch.isfinite(position_e).all()
        assert torch.isfinite(velocity_w).all()
        assert torch.isfinite(reward).all()

        x = position_e[0].item()
        y = position_e[1].item()
        vx = velocity_w[0].item()
        delta_vx = vx - launch_vx

        recovery = get_recovery_mask(env)[0].item()

        # 这里只记录球是否到过守门区域的前边界。
        reached_guard_region |= x <= settings.robot_x_limit
        crossed_reference |= x < 0.0
        recovery_seen |= recovery
        minimum_x = min(minimum_x, x)

        if (
            step == 0
            or (step + 1) % print_every == 0
            or terminated.any().item()
            or truncated.any().item()
        ):
            robot_z = (
                robot.data.root_link_pos_w[0, 2]
                - env.scene.env_origins[0, 2]
            ).item()

            print(
                f"t={(step + 1) * env.step_dt:5.2f}s "
                f"x={x:6.3f} y={y:6.3f} "
                f"vx={vx:6.3f} dvx={delta_vx:6.3f} "
                f"recovery={recovery} "
                f"robot_z={robot_z:.3f}"
            )

        if terminated.any().item() or truncated.any().item():
            break

    print("到过守门区域前边界:", reached_guard_region)
    print("球心经过 x=0:", crossed_reference)
    print("恢复门控曾开启:", recovery_seen)
    print("球的最小环境内 x:", minimum_x)
    print("完整来球过程记录完成；请根据输出判断轨迹")

finally:
    env.close()