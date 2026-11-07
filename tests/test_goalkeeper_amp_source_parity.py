"""Numerical comparison against the local source project, when available."""

import importlib.util
from pathlib import Path

import torch

from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.goalkeeper_normalizer import GoalkeeperAmpNormalizer


torch.set_num_threads(1)
source_root = (
    Path(__file__).resolve().parents[2] / "Humanoid-Goalkeeper" / "rsl_rl" / "rsl_rl"
)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source_parity():
    torch.manual_seed(42)
    if not (source_root / "modules/amp.py").exists():
        print("SKIP: local Humanoid-Goalkeeper checkout not available")
        return
    source_amp = load_source("source_goalkeeper_amp", source_root / "modules/amp.py")
    source_utils = load_source(
        "source_goalkeeper_utils", source_root / "utils/utils.py"
    )
    model = CategoricalAmpDiscriminator(empirical_normalization=False)
    discriminator = model.discriminators[0]
    assert discriminator.trunk[0].out_features == 512
    assert discriminator.trunk[2].out_features == 256
    # Same initial values, independently trainable parameter storage.
    for other in model.discriminators[1:]:
        for left, right in zip(discriminator.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            assert left.data_ptr() != right.data_ptr()
    for layer in (discriminator.trunk[2], discriminator.linear):
        assert layer.weight.abs().max().item() <= 1.0
        assert not layer.bias.any().item()

    reference = source_amp.AMP(44, 0.4, device="cpu")
    reference.trunk.load_state_dict(discriminator.trunk.state_dict())
    reference.amp_linear.load_state_dict(discriminator.linear.state_dict())
    expert, policy = torch.randn(6, 44), torch.randn(6, 44)
    classification, penalty = discriminator.compute_loss(
        discriminator(policy), discriminator(expert), expert, policy, lambda_=0.5
    )
    expected_loss, _, _ = reference.compute_loss(policy.clone(), expert.clone())
    torch.testing.assert_close(classification + penalty, expected_loss)
    (classification + penalty).backward()
    expected_loss.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, ours in discriminator.named_parameters():
        reference_name = name.replace("linear.", "amp_linear.")
        original = reference_parameters[reference_name]
        torch.testing.assert_close(ours.grad, original.grad, rtol=2e-5, atol=2e-5)

    shared = GoalkeeperAmpNormalizer(44)
    original_normalizer = source_utils.Normalizer(44)
    for raw in (expert, policy):
        normalized = shared(raw)
        original_normalized = original_normalizer.normalize_torch(raw, "cpu")
        torch.testing.assert_close(normalized, original_normalized)
        shared.update(normalized)
        original_normalizer.update(original_normalized.numpy())
    # NumPy/Torch batch reductions use float32; accumulated float64 buffers
    # therefore need a float32-sized tolerance, not float64 machine precision.
    torch.testing.assert_close(
        shared.mean, torch.from_numpy(original_normalizer.mean), rtol=2e-6, atol=3e-7
    )
    torch.testing.assert_close(
        shared.var, torch.from_numpy(original_normalizer.var), rtol=2e-6, atol=3e-7
    )
    assert shared.count.item() == original_normalizer.count

    normalized_model = CategoricalAmpDiscriminator()
    normalized_model.discriminators[0].load_state_dict(discriminator.state_dict())
    normalized_model.amp_normalizer.load_state_dict(shared.state_dict())
    # Use identical accumulated statistics to isolate the reward calculation.
    normalized_model.amp_normalizer.mean.copy_(
        torch.from_numpy(original_normalizer.mean)
    )
    normalized_model.amp_normalizer.var.copy_(torch.from_numpy(original_normalizer.var))
    ids = torch.zeros(6, dtype=torch.int64)
    torch.manual_seed(123)
    reward = normalized_model.predict_reward(policy, ids)
    torch.manual_seed(123)
    expected_reward = reference.predict_reward(policy, original_normalizer).squeeze(-1)
    torch.testing.assert_close(reward, expected_reward)
    print("原仓库数值对照：网络配置、loss、梯度、归一化和噪声奖励全部通过")


check_source_parity()
