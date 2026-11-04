import torch 

from booster_mjlab.amp.modules.history import (
    TwoFrameAmpHistory,
    select_valid_amp_samples,
)

from booster_mjlab.amp.modules.amp import AmpReplayBuffer


class CategoricalAmpReplayBuffer:
    def __init__(
        self,
        capacity_per_class: int,
        device: str | torch.device = "cpu",
    ):
        if capacity_per_class <= 0:
            raise ValueError("capacity_per_class must be a positive integer")

        self.device = torch.device(device)
        self._buffer = [
            AmpReplayBuffer(
                obs_dim=44,
                buffer_size=capacity_per_class,
                device=self.device,
            )
            for _ in range(6)
        ]

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(len(buffer) for buffer in self._buffer)

    def insert(
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
            raise ValueError("motion_ids must be int64 (B,) on buffer device")
        if ((motion_ids < 0) | (motion_ids > 5)).any().item():
            raise ValueError("motion_ids must be in [0, 5]")

        for motion_id, buffer in enumerate(self._buffer):
            mask = motion_ids == motion_id
            if mask.any().item():
                buffer.insert(features[mask].detach())

    def sample(
        self,
        motion_id: int,
        batch_size: int,
    ) -> torch.Tensor:
        if not isinstance(motion_id, int) or not 0 <= motion_id < 6:
            raise ValueError("motion_id must be an integer in [0, 5]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        buffer = self._buffer[motion_id]
        if len(buffer) == 0:
            raise ValueError(f"No samples available for motion_id {motion_id}")

        indices = torch.randint(
            len(buffer),
            (batch_size,),
            device=self.device,
        )

        return buffer.observations[indices]

        
