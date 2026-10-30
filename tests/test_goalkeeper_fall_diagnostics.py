import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)


cfg = booster_k1_goalkeeper_env_cfg(num_envs=1)
cfg.auto_reset = False

env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    env.reset(seed=42)

    robot = env.scene["robot"]
    feet = env.scene["feet_ground_contact"]
    joint_action = env.action_manager.get_term("joint_pos")

    joint_ids = joint_action.target_ids
    joint_names = list(joint_action.target_names)

    # 这里是 action 所选关节内部的列索引。
    leg_columns = [
        i
        for i, name in enumerate(joint_names)
        if any(part in name for part in ("_Hip_", "_Knee_", "_Ankle_"))
    ]
    assert len(leg_columns) == 12

    print("脚部数据列顺序:", feet.primary_names)
    print("受控关节顺序:", joint_names)

    initial_error = (
        robot.data.joint_pos[0, joint_ids]
        - robot.data.default_joint_pos[0, joint_ids]
    ).abs().max()

    print(
        "reset 后关节位置与默认位置的最大差值:",
        initial_error.item(),
        "rad",
    )

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    print_every = max(1, round(0.1 / env.step_dt))
    num_steps = min(
        round(3.0 / env.step_dt),
        env.max_episode_length,
    )

    for step in range(num_steps):
        _, reward, terminated, truncated, _ = env.step(action)

        if (
            step != 0
            and (step + 1) % print_every != 0
            and not terminated.any().item()
            and not truncated.any().item()
        ):
            continue

        gravity_b = robot.data.projected_gravity_b[0]
        angular_velocity_b = robot.data.root_link_ang_vel_b[0]

        height = (
            robot.data.root_link_pos_w[0, 2]
            - env.scene.env_origins[0, 2]
        ).item()

        cos_tilt = (-gravity_b[2]).clamp(-1.0, 1.0).item()
        tilt_deg = math.degrees(math.acos(cos_tilt))

        actual_q = robot.data.joint_pos[0, joint_ids]
        target_q = robot.data.joint_pos_target[0, joint_ids]

        expected_target = (
            robot.data.default_joint_pos[0, joint_ids]
            - robot.data.encoder_bias[0, joint_ids]
        )

        target_error = (target_q - expected_target).abs().max()
        tracking_error = actual_q - target_q

        leg_error = tracking_error[leg_columns]
        leg_rmse = leg_error.square().mean().sqrt()

        torque = robot.data.qfrc_actuator[0, joint_ids]

        found = feet.data.found
        force = feet.data.force
        assert found is not None
        assert force is not None

        for value in (
            gravity_b,
            angular_velocity_b,
            actual_q,
            target_q,
            torque,
            force,
            reward,
        ):
            assert torch.isfinite(value).all()

        print(
            f"\nt={(step + 1) * env.step_dt:.2f}s "
            f"height={height:.3f}m "
            f"tilt={tilt_deg:.1f}deg "
            f"g_b=({gravity_b[0].item():+.3f},"
            f"{gravity_b[1].item():+.3f},"
            f"{gravity_b[2].item():+.3f}) "
            f"omega_xy="
            f"{angular_velocity_b[:2].norm().item():.3f}rad/s"
        )

        for i, name in enumerate(feet.primary_names):
            foot_force = force[0, i]
            print(
                f"  {name}: "
                f"contact={found[0, i].item() > 0} "
                f"|F|={foot_force.norm().item():.1f}N "
                f"Fz={foot_force[2].item():+.1f}N"
            )

        print(
            f"  zero_target_error={target_error.item():.6f}rad "
            f"leg_tracking_rmse={leg_rmse.item():.4f}rad"
        )

        worst = leg_error.abs().topk(2).indices.tolist()

        for local_index in worst:
            column = leg_columns[local_index]
            print(
                f"  {joint_names[column]}: "
                f"q={actual_q[column].item():+.3f} "
                f"target={target_q[column].item():+.3f} "
                f"error={tracking_error[column].item():+.3f}rad "
                f"torque={torque[column].item():+.2f}Nm"
            )

        if terminated.any().item() or truncated.any().item():
            break

    print("\n倒地诊断记录完成；这不是站立稳定性通过判定")

finally:
    env.close()