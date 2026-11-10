from copy import deepcopy

import torch
from torch import nn

from booster_mjlab.amp.modules.discriminator import Discriminator
from booster_mjlab.amp.modules.history import select_valid_amp_samples
from booster_mjlab.amp.modules.goalkeeper_normalizer import GoalkeeperAmpNormalizer


class CategoricalAmpDiscriminator(nn.Module):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        *,
        empirical_normalization: bool = True,
        hidden_layer_sizes: tuple[int, ...] = (512, 256),
        logit_init_scale: float = 1.0,
        reward_mode: str = "source_noise",
    ):
        super().__init__()

        self.device = torch.device(device)
        if reward_mode not in ("source_noise", "direct"):
            raise ValueError("reward_mode must be 'source_noise' or 'direct'")
        self.reward_mode = reward_mode
        if not hidden_layer_sizes or any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive widths")
        if not 0.0 < logit_init_scale < float("inf"):
            raise ValueError("logit_init_scale must be finite and positive")
        self.amp_normalizer = (
            GoalkeeperAmpNormalizer(44).to(self.device)
            if empirical_normalization
            else None
        )
        prototype = Discriminator(
            input_dim=44,
            hidden_layer_sizes=list(hidden_layer_sizes),
            reward_scale=1.0,
            device=self.device,
            loss_type="least_squares",
            use_minibatch_std=False,
            empirical_normalization=False,
        )
        # Source leaves the first layer at PyTorch defaults and initializes
        # subsequent layers uniformly, then deep-copies the fresh prototype.
        later_layers = [
            layer for layer in prototype.trunk if isinstance(layer, nn.Linear)
        ][1:] + [prototype.linear]
        for layer in later_layers:
            nn.init.uniform_(layer.weight, -logit_init_scale, logit_init_scale)
            nn.init.zeros_(layer.bias)
        self.discriminators = nn.ModuleList([deepcopy(prototype) for _ in range(6)])

    @property
    def reward_version(self) -> str:
        """Checkpoint identity; unchanged source mode accepts old checkpoints."""
        return {
            "source_noise": "least_squares_noise20_sigma0.3",
            "direct": "least_squares_direct_v1",
        }[self.reward_mode]

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Use the shared normalizer without updating its statistics."""
        if self.amp_normalizer is None:
            return features
        return self.amp_normalizer(features)

    @torch.no_grad()
    def update_normalization(self, *normalized_batches: torch.Tensor) -> None:
        """Intentional source behavior: update using normalized, not raw data."""
        if self.amp_normalizer is not None:
            for batch in normalized_batches:
                self.amp_normalizer.update(batch.detach())

    def forward(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return discriminator scores, independent of the reward mode."""
        self._validate_inputs(features, motion_ids)

        features = self.normalize(features)

        scores = features.new_empty((features.shape[0], 1))

        for motion_id, discriminator in enumerate(self.discriminators):
            mask = motion_ids == motion_id
            if mask.any().item():
                scores[mask] = discriminator(features[mask])

        return scores

    @torch.no_grad()
    def predict_reward(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute style reward; direct mode never samples synthetic candidates."""
        self._validate_inputs(features, motion_ids)

        features = self.normalize(features)

        rewards = features.new_empty((features.shape[0],))

        for motion_id, discriminator in enumerate(self.discriminators):
            mask = motion_ids == motion_id
            if mask.any().item():
                if self.reward_mode == "direct":
                    # Features were normalized once above; sub-Ds have no
                    # normalizer. Evaluate only the actual detached trajectory.
                    scores = discriminator(features[mask]).squeeze(-1)
                    rewards[mask] = (1.0 - 0.25 * (scores - 1.0).square()).clamp_min(
                        0.0
                    )
                else:
                    rewards[mask] = discriminator.predict_reward(features[mask])

        return rewards

    def _validate_inputs(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != 44:
            raise ValueError("features must have shape (B, 44)")
        if features.dtype != torch.float32 or features.device != self.device:
            raise ValueError("features dtype/device mismatch")
        if (
            motion_ids.shape != (features.shape[0],)
            or motion_ids.dtype != torch.int64
            or motion_ids.device != self.device
        ):
            raise ValueError("motion_ids must be int64 (B,) on model device")
        if ((motion_ids < 0) | (motion_ids > 5)).any().item():
            raise ValueError("motion_ids must be in [0, 5]")


@torch.no_grad()
def predict_valid_amp_rewards(
    model: CategoricalAmpDiscriminator,
    windows: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_ids_before: torch.Tensor,
) -> torch.Tensor:
    features, motion_ids = select_valid_amp_samples(
        windows,
        valid_mask,
        motion_ids_before,
    )

    rewards = windows.new_zeros((windows.shape[0],))

    rewards[valid_mask] = model.predict_reward(features, motion_ids)

    return rewards
