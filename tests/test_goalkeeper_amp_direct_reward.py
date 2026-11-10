"""Direct AMP formula, shared normalization, routing and checkpoint boundaries."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mjlab  # noqa: F401 -- use the same bootstrap as training
import torch
from torch import nn

from goalkeeper_amp_test_utils import FakeEnv, make_runner
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
    predict_valid_amp_rewards,
)
from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.amp.runners.categorical_runner_cfg import CategoricalAmpRunnerCfg


class OffsetScore(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, features):
        return features[:, :1] + self.offset


model = CategoricalAmpDiscriminator(
    empirical_normalization=False,
    hidden_layer_sizes=(16, 8),
    reward_mode="direct",
)
model.discriminators = nn.ModuleList(OffsetScore(i) for i in range(6))
ids = torch.arange(6, dtype=torch.int64)
features = torch.zeros(6, 44)
features[:, 0] = torch.tensor([1.0, 0.0, -1.0, -2.0, 3.0, 1.0]) - ids.float()
features.requires_grad_()
rng = torch.get_rng_state().clone()
with patch("torch.randn", side_effect=AssertionError("Direct reward sampled noise")):
    reward = model.predict_reward(features, ids)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.75, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(model.predict_reward(features, ids), reward)
    assert model.predict_reward(features[:0], ids[:0]).shape == (0,)
    assert model.predict_reward(features[:1], ids[:1]).shape == (1,)
assert not reward.requires_grad
assert torch.equal(rng, torch.get_rng_state())
assert model.reward_version == "least_squares_direct_v1"

# Done rows remain zero; only valid windows enter the model.
valid = torch.tensor([False, True, True, False, True, True])
window_reward = predict_valid_amp_rewards(
    model, features.detach().reshape(6, 2, 22), valid, ids
)
torch.testing.assert_close(
    window_reward, torch.where(valid, reward, torch.zeros_like(reward))
)
print(
    "Direct reward: hand-computed formula, six-class routing, no RNG and done masking passed"
)

# A real shared normalizer must be applied exactly once, without changing stats.
torch.manual_seed(42)
normalized = CategoricalAmpDiscriminator(
    hidden_layer_sizes=(16, 8), reward_mode="direct"
)
with torch.no_grad():
    normalized.amp_normalizer.mean.fill_(0.2)
    normalized.amp_normalizer.var.fill_(0.5)
raw = torch.randn(6, 44)
stats_before = {k: v.clone() for k, v in normalized.amp_normalizer.state_dict().items()}
with torch.no_grad():
    expected = (
        1.0 - 0.25 * (normalized(raw, ids).squeeze(-1) - 1.0).square()
    ).clamp_min(0.0)
with patch.object(normalized, "normalize", wraps=normalized.normalize) as normalize_spy:
    torch.testing.assert_close(normalized.predict_reward(raw, ids), expected)
    normalize_spy.assert_called_once()
for k, v in normalized.amp_normalizer.state_dict().items():
    torch.testing.assert_close(v, stats_before[k], rtol=0, atol=0)

# Switching reward mode does not change raw scores, parameters or the source path.
source = CategoricalAmpDiscriminator(hidden_layer_sizes=(16, 8))
source.load_state_dict(normalized.state_dict())
torch.testing.assert_close(source(raw, ids), normalized(raw, ids))
torch.manual_seed(17)
source_reward = source.predict_reward(raw, ids)
torch.manual_seed(17)
expected_source = torch.empty(6)
source_features = source.normalize(raw)
for i, discriminator in enumerate(source.discriminators):
    expected_source[i] = discriminator.predict_reward(source_features[i : i + 1])[0]
torch.testing.assert_close(source_reward, expected_source)
assert source.reward_version == "least_squares_noise20_sigma0.3"
try:
    CategoricalAmpDiscriminator(hidden_layer_sizes=(8,), reward_mode="typo")
except ValueError:
    pass
else:
    raise AssertionError("Invalid reward mode was accepted")
print("Shared normalization, readonly statistics and source-noise compatibility passed")

# Legacy configuration dicts without the new optional field retain source mode.
legacy_cfg = CategoricalAmpRunnerCfg()
legacy_cfg.actor.hidden_dims = legacy_cfg.critic.hidden_dims = (16, 8)
legacy_cfg.amp.hidden_layer_sizes = (16, 8)
legacy_dict = asdict(legacy_cfg)
legacy_dict["amp"].pop("reward_mode")
legacy_runner = CategoricalAmpOnPolicyRunner(FakeEnv(), legacy_dict, device="cpu")
assert legacy_runner.amp_model.reward_mode == "source_noise"

with TemporaryDirectory(prefix="goalkeeper_amp_modes_") as directory:
    direct, _ = make_runner(reward_mode="direct")
    direct.learn(1)
    assert 0.0 <= direct._last_iteration_summary["reward_means"]["amp_raw"] <= 1.0
    checkpoint = Path(directory) / "direct.pt"
    direct.save(str(checkpoint))
    restored, _ = make_runner(reward_mode="direct")
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 1
    for k, v in direct.amp_model.state_dict().items():
        torch.testing.assert_close(
            restored.amp_model.state_dict()[k], v, rtol=0, atol=0
        )
    restored.learn(1)
    assert restored.current_learning_iteration == 2
    assert all(torch.isfinite(p).all() for p in restored.amp_model.parameters())

    original, _ = make_runner()
    original.learn(1)
    old_checkpoint = Path(directory) / "source.pt"
    original.save(str(old_checkpoint))
    for mode, path in (("direct", old_checkpoint), ("source_noise", checkpoint)):
        wrong_mode, _ = make_runner(reward_mode=mode)
        actor_before = {
            k: v.clone() for k, v in wrong_mode.alg.actor.state_dict().items()
        }
        try:
            wrong_mode.load(str(path))
        except ValueError as error:
            assert "reward_version" in str(error)
        else:
            raise AssertionError("Full resume silently changed reward semantics")
        assert wrong_mode.current_learning_iteration == 0
        assert not wrong_mode.alg.optimizer.state
        for k, v in wrong_mode.alg.actor.state_dict().items():
            torch.testing.assert_close(v, actor_before[k], rtol=0, atol=0)

    # Explicit warm start may load Actor/Critic across modes, not old D/Adam.
    warm, _ = make_runner(reward_mode="direct", warm_start=True)
    amp_before = {k: v.clone() for k, v in warm.amp_model.state_dict().items()}
    warm.load(str(old_checkpoint))
    assert warm.current_learning_iteration == warm.total_timesteps == 0
    assert not warm.alg.optimizer.state
    for k, v in warm.amp_model.state_dict().items():
        torch.testing.assert_close(v, amp_before[k], rtol=0, atol=0)
    for k, v in original.alg.actor.state_dict().items():
        torch.testing.assert_close(warm.alg.actor.state_dict()[k], v, rtol=0, atol=0)

print(
    "Direct PPO update/resume, cross-mode rejection, legacy config and warm start passed"
)
