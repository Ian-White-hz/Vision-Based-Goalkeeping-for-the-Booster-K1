import torch

from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
    predict_valid_amp_rewards,
)


model = CategoricalAmpDiscriminator(
    empirical_normalization=True,
)
model.train()

# 设置测试分数，并初始化每个判别器的统计量。
with torch.no_grad():
    for motion_id, discriminator in enumerate(model.discriminators):
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(motion_id))

    model.update_normalization(torch.zeros(2, 44))

state_before = {name: value.clone() for name, value in model.state_dict().items()}

windows = torch.randn(3, 2, 22, requires_grad=True)
valid_mask = torch.tensor([True, False, True])
motion_ids_before = torch.tensor([0, 3, 5], dtype=torch.int64)

rewards = predict_valid_amp_rewards(
    model,
    windows,
    valid_mask,
    motion_ids_before,
)

expected = torch.tensor(
    [
        0.75,  # 类别 0：固定输出 0。
        0.0,  # 无效窗口，不计算 AMP 奖励。
        0.0,  # 类别 5：固定输出 5，平方奖励被截断为 0。
    ]
)

assert rewards.shape == (3,)
torch.testing.assert_close(rewards, expected)
assert not rewards.requires_grad
assert windows.grad is None
assert all(parameter.grad is None for parameter in model.parameters())

# 全部无效时，返回每个环境的零奖励。
zero_rewards = predict_valid_amp_rewards(
    model,
    windows,
    torch.zeros(3, dtype=torch.bool),
    motion_ids_before,
)
torch.testing.assert_close(zero_rewards, torch.zeros(3))

# 参数和归一化统计都不得改变。
for name, value in model.state_dict().items():
    torch.testing.assert_close(
        value,
        state_before[name],
        rtol=0,
        atol=0,
    )

print("当前环境 AMP 奖励:", rewards)
print("分类奖励、无效窗口与状态不变检查通过")
