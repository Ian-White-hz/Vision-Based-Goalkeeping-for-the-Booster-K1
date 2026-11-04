import torch

from booster_mjlab.amp.modules.history import (
    TwoFrameAmpHistory,
    select_valid_amp_samples
)


class CategoricalAmpCollector:
    def __init__(
        self,
        model,
        replay,
        initial_q: torch.Tensor,
    ):
        if model.device != replay.device:
            raise ValueError(
                f"Model device {model.device} does not match replay device {replay.device}"
            )
        if initial_q.device != model.device:
            raise ValueError(
                f"Initial_q device {initial_q.device} does not match model device {model.device}"
            )

        self.model = model
        self.replay = replay
        self.history = TwoFrameAmpHistory(initial_q)

        self.num_envs = initial_q.shape[0]
        self._motion_ids_before: torch.Tensor | None = None

    def begin_step(
        self,
        motion_ids: torch.Tensor,
    ) -> None:
        if self._motion_ids_before is not None:
            raise RuntimeError("begin_step called before end_step")

        if (
            motion_ids.shape != (self.num_envs,)
            or motion_ids.dtype != torch.int64
            or motion_ids.device != self.model.device
        ):
            raise ValueError(
                f"motion_ids must be int64 ({self.num_envs},) on model device {self.model.device}"
            )

        if ((motion_ids < 0) | (motion_ids > 5)).any().item():
            raise ValueError("motion_ids must be in [0, 5]")

        self._motion_ids_before = motion_ids.clone()

    @torch.no_grad()
    def finish_step(
        self,
        next_q: torch.Tensor,
        dones: torch.Tensor,
    )   -> torch.Tensor:
        if self._motion_ids_before is None:
            raise RuntimeError("Call begin_step before finish_step")

        windows, valid_mask = self.history.push(next_q, dones)

        features, motion_ids = select_valid_amp_samples(
            windows,
            valid_mask,
            self._motion_ids_before,
        )

        amp_rewards = next_q.new_zeros((self.num_envs,))
        amp_rewards[valid_mask] = self.model.predict_reward(
            features,
            motion_ids
        )

        self.replay.insert(features, motion_ids)

        self._motion_ids_before = None
        return amp_rewards