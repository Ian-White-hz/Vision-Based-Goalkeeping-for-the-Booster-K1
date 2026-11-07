import torch

from booster_mjlab.amp.modules.discriminator import Discriminator
from booster_mjlab.amp.modules.losses import make_loss_function


# 1. 验证最小二乘分类 loss。
loss_fn = make_loss_function("least_squares")

loss = loss_fn(
    policy_samples=torch.tensor([[-1.0], [0.0]]),
    expert_samples=torch.tensor([[1.0], [0.0]]),
)

# expert: (0 + 1) / 2
# policy: (0 + 1) / 2
torch.testing.assert_close(loss, torch.tensor(1.0))


# 2. 构造可手算的判别器。
model = Discriminator(
    input_dim=2,
    hidden_layer_sizes=[2],
    reward_scale=1.0,
    loss_type="least_squares",
    use_minibatch_std=False,
    empirical_normalization=False,
)

with torch.no_grad():
    model.trunk[0].weight.copy_(torch.eye(2))
    model.trunk[0].bias.zero_()
    model.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))
    model.linear.bias.zero_()

# 所有输入为正，因此 ReLU 不改变输入：
# D(x) = x1 + 2*x2，梯度为 [1, 2]。
expert = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
policy = torch.tensor([[0.2, 0.4], [0.3, 0.5]])

classification_loss, gradient_penalty = model.compute_loss(
    policy_d=model(policy),
    expert_d=model(expert),
    sample_amp_expert=expert,
    sample_amp_policy=policy,
    lambda_=0.5,
)

expected_classification = (
    (model(expert) - 1.0).square().mean()
    + (model(policy) + 1.0).square().mean()
)

torch.testing.assert_close(
    classification_loss,
    expected_classification,
)
torch.testing.assert_close(
    gradient_penalty,
    torch.tensor(2.5),
)

(classification_loss + gradient_penalty).backward()
assert model.linear.weight.grad is not None
assert torch.isfinite(model.linear.weight.grad).all().item()

print("原版分类 loss 与梯度惩罚检查通过")


# 3. 输出固定分数，奖励可以手算。
with torch.no_grad():
    model.linear.weight.zero_()

for score, expected in [
    (-3.0, 0.0),
    (-1.0, 0.0),
    (0.0, 0.75),
    (1.0, 1.0),
    (3.0, 0.0),
]:
    with torch.no_grad():
        model.linear.bias.fill_(score)

    rewards = model.predict_reward(torch.zeros(2, 2))

    assert rewards.shape == (2,)
    assert not rewards.requires_grad

    torch.testing.assert_close(
        rewards,
        torch.full((2,), expected),
    )

print("原版固定分数奖励检查通过")


# 4. 固定随机种子，独立计算噪声候选的预期奖励。
with torch.no_grad():
    model.linear.weight.copy_(torch.tensor([[0.2, 0.1]]))
    model.linear.bias.zero_()

features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

torch.manual_seed(123)
noise = torch.randn(2, 20, 2) * 0.3
candidates = features[:, None, :] + noise

# 当前 trunk 是单位矩阵 + ReLU。
scores = (
    candidates.relu() * torch.tensor([0.2, 0.1])
).sum(dim=-1)

expected_rewards = (
    1.0 - 0.25 * (scores - 1.0).square().min(dim=1).values
).clamp_min(0.0)

torch.manual_seed(123)
actual_rewards = model.predict_reward(features)

torch.testing.assert_close(actual_rewards, expected_rewards)

print("原版噪声采样奖励检查通过")