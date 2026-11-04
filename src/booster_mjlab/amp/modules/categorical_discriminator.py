import torch
from torch import nn

from booster_mjlab.amp.modules.discriminator import Discriminator
from booster_mjlab.amp.modules.history import select_valid_amp_samples


class CategoricalAmpDiscriminator(nn.Module):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        *,
        empirical_normalization: bool = False,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.discriminators = nn.ModuleList([
            Discriminator(
                input_dim=44,
                hidden_layer_sizes=[64, 32],
                reward_scale=1.0,
                device=self.device,
                loss_type="hinge",
                use_minibatch_std=False,
                empirical_normalization=empirical_normalization,
            )
            for _ in range(6)
        ])

    def forward(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(features, motion_ids)

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
        self._validate_inputs(features, motion_ids)

        rewards = features.new_empty((features.shape[0],))

        for motion_id, discriminator in enumerate(self.discriminators):
            mask = motion_ids == motion_id
            if mask.any().item():
                rewards[mask] = discriminator.predict_reward(features[mask])

        return rewards
    
    def _validate_inputs(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != 44:
            raise ValueError("features must have shape (B, 44)")
        if (
            features.dtype != torch.float32
            or features.device != self.device
        ):
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