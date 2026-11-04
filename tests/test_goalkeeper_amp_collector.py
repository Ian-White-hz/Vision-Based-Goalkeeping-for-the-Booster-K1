import torch
import torch.nn.functional as F

from booster_mjlab.amp.modules.categorical_collector import (
    CategoricalAmpCollector,
)
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.categorical_replay import (
    CategoricalAmpReplayBuffer,
)


model = CategoricalAmpDiscriminator()
replay = CategoricalAmpReplayBuffer(capacity_per_class=8)

with torch.no_grad():
    for motion_id, discriminator in enumerate(model.discriminators):
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(motion_id))

q0 = torch.stack([
    torch.full((22,), 0.0),
    torch.full((22,), 0.5),
])

collector = CategoricalAmpCollector(model, replay, q0)
motion_ids = torch.tensor([0, 3], dtype=torch.int64)

# 第一步：先保存类别，再模拟环境执行和自动 reset。
collector.begin_step(motion_ids)

motion_ids[1] = 5
q1 = torch.stack([
    torch.full((22,), 0.1),
    torch.full((22,), -0.2),
])

rewards1 = collector.finish_step(
    q1,
    torch.tensor([False, True]),
)

r0 = F.softplus(torch.tensor(0.0))
r5 = F.softplus(torch.tensor(5.0))

torch.testing.assert_close(
    rewards1,
    torch.stack([r0, torch.tensor(0.0)]),
)
assert replay.counts == (1, 0, 0, 0, 0, 0)

# 第二步：环境 1 已进入类别 5 的新回合。
collector.begin_step(motion_ids)

q2 = torch.stack([
    torch.full((22,), 0.2),
    torch.full((22,), -0.1),
])

rewards2 = collector.finish_step(
    q2,
    torch.tensor([False, False]),
)

torch.testing.assert_close(rewards2, torch.stack([r0, r5]))
assert replay.counts == (2, 0, 0, 0, 0, 1)

sample5 = replay.sample(motion_id=5, batch_size=1)
torch.testing.assert_close(sample5[0, :22], q1[1])
torch.testing.assert_close(sample5[0, 22:], q2[1])

assert not rewards1.requires_grad
assert not rewards2.requires_grad

print("第一步奖励:", rewards1)
print("第二步奖励:", rewards2)
print("Replay 数量:", replay.counts)
print("AMP 单步收集链路检查通过")