import torch

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards

task = torch.tensor([0.2, -0.4])
amp = torch.tensor([0.75, 1.0])
original_task, original_amp = task.clone(), amp.clone()
expected = 0.6 * task + 0.4 * amp
for dt in (0.02, 0.01):
    for scale_by_dt in (True, False):
        total, task_part, style_part = mix_amp_rewards(
            task,
            amp,
            style_weight=0.4,
            step_dt=dt,
            scale_rewards_by_dt=scale_by_dt,
        )
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(task_part, 0.6 * task)
        torch.testing.assert_close(style_part, 0.4 * amp)
torch.testing.assert_close(task, original_task)
torch.testing.assert_close(amp, original_amp)
for weight, target in ((0.0, task), (1.0, amp)):
    total, _, _ = mix_amp_rewards(
        task, amp, style_weight=weight, step_dt=0.02, scale_rewards_by_dt=True
    )
    torch.testing.assert_close(total, target)
assert not total.requires_grad
for weight in (-0.1, 1.1, float("nan")):
    try:
        mix_amp_rewards(
            task, amp, style_weight=weight, step_dt=0.02, scale_rewards_by_dt=True
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid weight accepted")
print("原版奖励直接混合、无额外 dt、边界权重和输入不变检查通过")
