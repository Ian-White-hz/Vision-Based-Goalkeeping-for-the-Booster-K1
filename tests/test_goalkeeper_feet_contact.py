import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)


cfg = booster_k1_goalkeeper_env_cfg(num_envs=1)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    env.reset(seed=42)

    reward_cfg = env.reward_manager.get_term_cfg("feet_slippage")
    sharp_cfg = env.reward_manager.get_term_cfg(
        "penalize_sharpcontact"
    )

    sensor = env.scene["feet_ground_contact"]
    assert isinstance(sensor, ContactSensor)

    foot_names = sensor.primary_names
    print("传感器对应的脚:", foot_names)

    assert len(foot_names) == 2
    assert set(foot_names) == {
        "left_foot_link",
        "right_foot_link",
    }

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    contact_seen = torch.zeros(
        (env.num_envs, 2),
        dtype=torch.bool,
        device=env.device,
    )

    for _ in range(20):
        _, total_reward, _, _, _ = env.step(action)

        assert torch.isfinite(total_reward).all()

        raw_reward = reward_cfg.func(env, **reward_cfg.params)

        assert raw_reward.shape == (env.num_envs,)
        assert torch.isfinite(raw_reward).all()
        assert (raw_reward >= 0.0).all()
        assert (raw_reward <= 1.0).all()

        found = sensor.data.found
        force_w = sensor.data.force

        assert found is not None
        assert force_w is not None

        sharp_cost = sharp_cfg.func(env, **sharp_cfg.params)

        assert sharp_cost.shape == (env.num_envs,)
        assert torch.isfinite(sharp_cost).all()
        assert ((sharp_cost == 0.0) | (sharp_cost == 1.0)).all()

        expected_cost = (
            torch.linalg.vector_norm(force_w, dim=-1).mean(dim=1)
            > sharp_cfg.params["max_contact_force"]
        ).float()

        torch.testing.assert_close(sharp_cost, expected_cost)

        assert found.shape == (1, 2)
        assert force_w.shape == (1, 2, 3)

        assert torch.isfinite(found).all()
        assert torch.isfinite(force_w).all()

        contact_seen |= found > 0

    print("脚部过大接触力原始代价:", sharp_cost)

    print("脚部防滑原始奖励:", raw_reward)
    print("脚部防滑 reward 接入检查通过")

    print("是否曾检测到接触:", contact_seen)
    print("当前接触数量:", sensor.data.found)
    print("当前世界坐标系接触力:", sensor.data.force)

    assert contact_seen.all(), "短时测试中有脚始终未检测到地面接触"

    print("双脚地面接触传感器检查通过")

finally:
    env.close()