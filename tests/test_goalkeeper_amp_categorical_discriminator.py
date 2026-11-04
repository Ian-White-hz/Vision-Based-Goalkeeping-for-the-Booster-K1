import torch

from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)


model = CategoricalAmpDiscriminator()

with torch.no_grad():
    for motion_id, discriminator in enumerate(model.discriminators):
        # 最后一层权重为零，输出就只由 bias 决定。
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(motion_id))

features = torch.randn(5, 44)
motion_ids = torch.tensor([5, 0, 2, 5, 1], dtype=torch.int64)

scores = model(features, motion_ids)

assert scores.shape == (5, 1)
torch.testing.assert_close(
    scores,
    motion_ids.to(torch.float32).unsqueeze(1),
)

# 本批没有类别 3，修改 D_3 不应影响结果。
with torch.no_grad():
    model.discriminators[3].linear.bias.fill_(100.0)

torch.testing.assert_close(
    model(features, motion_ids),
    scores,
)

# 没有有效样本时，返回空结果。
empty_scores = model(
    torch.empty((0, 44)),
    torch.empty((0,), dtype=torch.int64),
)
assert empty_scores.shape == (0, 1)

# 六个网络都应注册在外层模型中。
state = model.state_dict()
for motion_id in range(6):
    assert f"discriminators.{motion_id}.linear.weight" in state

print("输入类别:", motion_ids)
print("输出分数:", scores.detach().squeeze(-1))
print("六判别器类别路由检查通过")



# Step 2: 测试训练时的梯度隔离
torch.manual_seed(42)

training_model = CategoricalAmpDiscriminator()
training_model.train()

optimizer = torch.optim.Adam(
    training_model.parameters(),
    lr=1.0e-3,
)


def check_one_class_update(target_id: int) -> None:
    expert_input = torch.full((8, 44), 0.2)
    policy_input = torch.full((8, 44), -0.2)
    ids = torch.full((8,), target_id, dtype=torch.int64)

    # 保存所有判别器更新前的参数。
    parameters_before = [
        [
            parameter.detach().clone()
            for parameter in discriminator.parameters()
        ]
        for discriminator in training_model.discriminators
    ]

    optimizer.zero_grad(set_to_none=True)

    # 通过容器路由，验证梯度也能穿过 scores[mask] 的写回操作。
    expert_scores = training_model(expert_input, ids)
    policy_scores = training_model(policy_input, ids)

    discriminator = training_model.discriminators[target_id]

    classification_loss, gradient_penalty = discriminator.compute_loss(
        policy_d=policy_scores,
        expert_d=expert_scores,
        sample_amp_expert=expert_input,
        sample_amp_policy=policy_input,
        lambda_=10.0,
    )

    total_loss = classification_loss + gradient_penalty
    assert torch.isfinite(total_loss).item()

    total_loss.backward()

    for motion_id, network in enumerate(training_model.discriminators):
        gradients = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.grad is not None
        ]

        if motion_id == target_id:
            assert gradients
            assert all(
                torch.isfinite(grad).all().item()
                for grad in gradients
            )
            assert sum(
                grad.square().sum().item()
                for grad in gradients
            ) > 0.0
        else:
            assert not gradients, f"D_{motion_id} received unexpected gradients"

    optimizer.step()

    changed = []

    for before, network in zip(
        parameters_before,
        training_model.discriminators,
    ):
        network_changed = any(
            not torch.equal(old, parameter.detach())
            for old, parameter in zip(before, network.parameters())
        )
        changed.append(network_changed)

    expected_changed = [
        motion_id == target_id
        for motion_id in range(6)
    ]
    assert changed == expected_changed

    print(f"训练类别 {target_id}，参数变化:", changed)
    print("total_loss:", total_loss.detach().item())


check_one_class_update(target_id=3)
check_one_class_update(target_id=1)

print("指定类别更新与未选网络隔离检查通过")