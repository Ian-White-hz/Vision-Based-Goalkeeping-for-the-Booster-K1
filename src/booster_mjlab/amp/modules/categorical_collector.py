import torch

from booster_mjlab.amp.modules.history import (
    TwoFrameAmpHistory,
    select_valid_amp_samples,
)


class CategoricalAmpCollector:
    """Build two-frame windows; no cross-rollout replay is used."""

    def __init__(self, model, initial_q: torch.Tensor):
        if initial_q.device != model.device:
            raise ValueError("initial_q must be on the model device")
        self.model = model
        self.history = TwoFrameAmpHistory(initial_q)
        self.num_envs = initial_q.shape[0]
        self._motion_ids_before = None
        self.last_features = None
        self.last_motion_ids = None
        self.last_valid_mask = None

    def begin_step(self, motion_ids: torch.Tensor) -> None:
        if self._motion_ids_before is not None:
            raise RuntimeError("begin_step called before finish_step")
        if (
            motion_ids.shape != (self.num_envs,)
            or motion_ids.dtype != torch.int64
            or motion_ids.device != self.model.device
        ):
            raise ValueError("motion_ids must be int64 (N,) on model device")
        if ((motion_ids < 0) | (motion_ids > 5)).any().item():
            raise ValueError("motion_ids must be in [0, 5]")
        self._motion_ids_before = motion_ids.clone()

    @torch.no_grad()
    def finish_step(self, next_q: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        if self._motion_ids_before is None:
            raise RuntimeError("Call begin_step before finish_step")
        windows, valid_mask = self.history.push(next_q, dones)
        features, ids = select_valid_amp_samples(
            windows, valid_mask, self._motion_ids_before
        )
        rewards = next_q.new_zeros(self.num_envs)
        rewards[valid_mask] = self.model.predict_reward(features, ids)

        # Preserve one row per PPO transition, including masked done rows.
        self.last_features = windows.flatten(start_dim=1).detach()
        self.last_motion_ids = self._motion_ids_before
        self.last_valid_mask = valid_mask
        self._motion_ids_before = None
        return rewards
