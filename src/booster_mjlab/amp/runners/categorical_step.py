import torch

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards


@torch.no_grad()
def collect_categorical_amp_step(
    env,
    alg,
    collector,
    obs,
    read_motion_ids,
    *,
    style_weight: float,
):
    device = collector.model.device

    if torch.device(alg.device) != device:
        raise ValueError(
            f"Algorithm device {alg.device} does not match collector device {device}"
        )

    obs = obs.to(device)

    motion_ids = read_motion_ids(env.unwrapped).to(device)
    collector.begin_step(motion_ids)

    actions = alg.act(obs)

    next_obs, task_rewards, dones, extras = env.step(
        actions.to(env.device),  # 将动作移动到环境设备上
    )

    next_obs = next_obs.to(device)
    task_rewards = task_rewards.to(device)
    dones = dones.to(device)

    amp_rewards = collector.finish_step(
        next_obs["amp"],
        dones.bool()
    )

    total_rewards, task_component, style_component = mix_amp_rewards(
        task_rewards,
        amp_rewards,
        style_weight=style_weight,
        step_dt=env.unwrapped.step_dt,
        scale_rewards_by_dt=env.cfg.scale_rewards_by_dt,
    )

    alg.process_env_step(
        next_obs,
        total_rewards,
        dones,
        extras,
    )

    reward_metrics = {
        "amp_raw": amp_rewards,
        "task_component": task_component,
        "style_component": style_component,
    }

    return next_obs, total_rewards, dones, extras, reward_metrics