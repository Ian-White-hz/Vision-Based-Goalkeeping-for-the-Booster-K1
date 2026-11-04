import torch

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards


step_dt = 0.02

# 假设环境内部的未缩放任务奖励是这些值。
raw_task_rewards = torch.tensor([10.0, -20.0])

# 模拟 scale_rewards_by_dt=True 的 env.step() 返回值。
task_rewards = raw_task_rewards * step_dt
amp_rewards = torch.tensor([3.0, 4.0])

task_snapshot = task_rewards.clone()
amp_snapshot = amp_rewards.clone()

total, task_part, style_part = mix_amp_rewards(
    task_rewards,
    amp_rewards,
    style_weight=0.4,
    step_dt=step_dt,
    scale_rewards_by_dt=True,
)

torch.testing.assert_close(task_part, torch.tensor([0.12, -0.24]))
torch.testing.assert_close(style_part, torch.tensor([0.024, 0.032]))
torch.testing.assert_close(total, torch.tensor([0.144, -0.208]))

# 输入不能被原地修改。
torch.testing.assert_close(task_rewards, task_snapshot)
torch.testing.assert_close(amp_rewards, amp_snapshot)

# 关闭 dt 缩放时，环境应返回未缩放的任务奖励。
unscaled_total, _, _ = mix_amp_rewards(
    raw_task_rewards,
    amp_rewards,
    style_weight=0.4,
    step_dt=step_dt,
    scale_rewards_by_dt=False,
)

torch.testing.assert_close(total, unscaled_total * step_dt)

# 权重为零：保留环境返回的完整任务奖励。
task_only, _, zero_style = mix_amp_rewards(
    task_rewards,
    amp_rewards,
    style_weight=0.0,
    step_dt=step_dt,
    scale_rewards_by_dt=True,
)

torch.testing.assert_close(task_only, task_rewards)
torch.testing.assert_close(zero_style, torch.zeros_like(task_rewards))

# 权重为一：只保留缩放后的 AMP 奖励。
style_only, zero_task, _ = mix_amp_rewards(
    task_rewards,
    amp_rewards,
    style_weight=1.0,
    step_dt=step_dt,
    scale_rewards_by_dt=True,
)

torch.testing.assert_close(style_only, amp_rewards * step_dt)
torch.testing.assert_close(zero_task, torch.zeros_like(task_rewards))

assert total.shape == (2,)
assert not total.requires_grad

print("任务分量:", task_part)
print("AMP 分量:", style_part)
print("总奖励:", total)
print("奖励组合与 dt 缩放检查通过")