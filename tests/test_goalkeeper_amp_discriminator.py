import torch

from booster_mjlab.amp.modules.discriminator import Discriminator


torch.manual_seed(42)

batch_size = 4
joint_dim = 22

# 构造两帧假关节角，只用于验证数据流。
q_previous = 0.1 * torch.randn(batch_size, joint_dim)
q_current = q_previous + 0.01 * torch.randn(batch_size, joint_dim)

# dim=1 新增时间维，顺序为：旧帧、新帧。
history = torch.stack(
    [q_previous, q_current],
    dim=1,
)
amp_input = history.reshape(batch_size, -1)

assert history.shape == (4, 2, 22)
assert amp_input.shape == (4, 44)

# 确认展平后没有把时间与关节顺序弄反。
torch.testing.assert_close(amp_input[:, :22], q_previous)
torch.testing.assert_close(amp_input[:, 22:], q_current)

discriminator = Discriminator(
    input_dim=44,
    hidden_layer_sizes=[64, 32],
    reward_scale=1.0,
    device=torch.device("cpu"),
    loss_type="hinge",
    use_minibatch_std=False,
    empirical_normalization=False,
)

scores = discriminator(amp_input)
rewards = discriminator.predict_reward(amp_input)

assert scores.shape == (batch_size, 1)
assert rewards.shape == (batch_size,)
assert torch.isfinite(scores).all()
assert torch.isfinite(rewards).all()

# 当前 hinge 模式的奖励转换是 softplus。
expected_rewards = torch.nn.functional.softplus(
    scores.detach()
).squeeze(-1)

torch.testing.assert_close(rewards, expected_rewards)

# 奖励作为数值交给 PPO，不保留判别器的梯度图。
assert not rewards.requires_grad

print("history:", tuple(history.shape))
print("amp_input:", tuple(amp_input.shape))
print("scores:", scores.detach())
print("rewards:", rewards)
print("判别器前向计算检查通过")


# 两类假运动，每行仍表示两帧展平后的 44 维特征。
expert_input = torch.full((8, 44), 0.2)
policy_input = torch.full((8, 44), -0.2)

optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=1.0e-3,
)

discriminator.train()
optimizer.zero_grad(set_to_none=True)

expert_scores = discriminator(expert_input)
policy_scores = discriminator(policy_input)

# 注意现有接口的参数顺序：policy 在前，expert 在后。
loss = discriminator.loss_fn(
    policy_scores,
    expert_scores,
)

# 手动计算公式，检查调用得到的 loss 是否一致。
expected_loss = 0.5 * (
    torch.relu(1.0 - expert_scores).mean()
    + torch.relu(1.0 + policy_scores).mean()
)

torch.testing.assert_close(loss, expected_loss)
assert loss.ndim == 0
assert torch.isfinite(loss).item()
assert loss.requires_grad

# 保存更新前的参数。clone 避免与原参数共享底层数据。
parameters_before = [
    parameter.detach().clone()
    for parameter in discriminator.parameters()
]

loss.backward()

# 验证产生了有限且非零的梯度。
gradients = [
    parameter.grad
    for parameter in discriminator.parameters()
    if parameter.grad is not None
]

assert gradients
assert all(torch.isfinite(grad).all().item() for grad in gradients)
assert sum(grad.square().sum().item() for grad in gradients) > 0.0

optimizer.step()

# 至少有一个参数张量发生变化。
parameter_changed = any(
    not torch.equal(before, after.detach())
    for before, after in zip(
        parameters_before,
        discriminator.parameters(),
    )
)
assert parameter_changed

print("expert 平均分数:", expert_scores.detach().mean().item())
print("policy 平均分数:", policy_scores.detach().mean().item())
print("更新前 hinge loss:", loss.detach().item())
print("参数是否更新:", parameter_changed)
print("判别器反向传播与参数更新检查通过")



## Step 3: 计算联合损失（分类 + 梯度惩罚）并反向传播
optimizer.zero_grad(set_to_none=True)

# 参数已经更新过，重新计算分数。
expert_scores = discriminator(expert_input)
policy_scores = discriminator(policy_input)

classification_loss, gradient_penalty = discriminator.compute_loss(
    policy_d=policy_scores,
    expert_d=expert_scores,
    sample_amp_expert=expert_input,
    sample_amp_policy=policy_input,
    lambda_=10.0,
)

total_loss = classification_loss + gradient_penalty

assert classification_loss.ndim == 0
assert gradient_penalty.ndim == 0
assert torch.isfinite(total_loss).item()
assert gradient_penalty.item() >= 0.0
assert gradient_penalty.requires_grad

# 当前关闭了归一化和 minibatch std，hinge 的 R1 计算是确定性的。
# 网络参数不变时，系数翻倍，惩罚也应该翻倍。
double_penalty = discriminator.compute_grad_penalty(
    expert_input,
    policy_input,
    lambda_=20.0,
)

torch.testing.assert_close(
    double_penalty,
    2.0 * gradient_penalty,
)

# 总损失反向传播，包含分类项和正则项。
total_loss.backward()

gradients = [
    parameter.grad
    for parameter in discriminator.parameters()
    if parameter.grad is not None
]

assert gradients
assert all(torch.isfinite(grad).all().item() for grad in gradients)

optimizer.step()

print("classification_loss:", classification_loss.detach().item())
print("gradient_penalty:", gradient_penalty.detach().item())
print("total_loss:", total_loss.detach().item())
print("梯度惩罚与联合更新检查通过")


## Step 4: 检查归一化器的累计统计量与单次归一化结果一致
normalized_discriminator = Discriminator(
    input_dim=44,
    hidden_layer_sizes=[64, 32],
    reward_scale=1.0,
    device=torch.device("cpu"),
    loss_type="hinge",
    use_minibatch_std=False,
    empirical_normalization=True,
)

normalized_discriminator.train()

# 两类来源共同更新同一个 normalizer。
normalized_discriminator.update_normalization(
    expert_input,
    policy_input,
)

normalizer = normalized_discriminator.amp_normalizer
assert normalizer is not None

raw_batch = torch.cat(
    [expert_input, policy_input],
    dim=0,
)

# 检查累计统计量与整批数据的计算结果一致。
expected_mean = raw_batch.mean(dim=0)
expected_std = raw_batch.std(dim=0, unbiased=False)

torch.testing.assert_close(normalizer.mean, expected_mean)
torch.testing.assert_close(normalizer.std, expected_std)
assert normalizer.count.item() == raw_batch.shape[0]

normalized_discriminator.eval()
count_before = normalizer.count.item()

with torch.no_grad():
    # 实际路径：原始输入 → 内部归一化 → 网络。
    actual_scores = normalized_discriminator(raw_batch)

    # 手动参考路径：只归一化一次，再直接经过 MLP。
    normalized_batch = (
        raw_batch - normalizer.mean
    ) / (normalizer.std + normalizer.eps)

    expected_scores = normalized_discriminator.linear(
        normalized_discriminator.trunk(normalized_batch)
    )

torch.testing.assert_close(actual_scores, expected_scores)
assert torch.isfinite(actual_scores).all()
assert normalizer.count.item() == count_before

print("normalizer 样本数:", normalizer.count.item())
print("mean 前三维:", normalizer.mean[:3])
print("std 前三维:", normalizer.std[:3])
print("共享归一化与单次归一化检查通过")