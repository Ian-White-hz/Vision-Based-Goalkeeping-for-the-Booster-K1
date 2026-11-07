"""Source Goalkeeper Normalizer rules, with checkpointable Torch buffers."""

import torch
from torch import nn


class GoalkeeperAmpNormalizer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.epsilon = 1.0e-4
        self.clip_obs = 10.0
        self.register_buffer("mean", torch.zeros(input_dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(input_dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1.0e-4, dtype=torch.float64))

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected shape (B, {self.input_dim})")
        if x.dtype != torch.float32 or x.device != self.mean.device:
            raise ValueError("Expected float32 input on the normalizer device")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        # Source casts var + epsilon before taking sqrt.
        mean = self.mean.to(dtype=x.dtype)
        std = (self.var + self.epsilon).to(dtype=x.dtype).sqrt()
        return ((x - mean) / std).clamp(-self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge supplied moments; the caller chooses raw vs normalized data."""
        self._validate_input(x)
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(dim=0).to(torch.float64)
        batch_var = x.var(dim=0, unbiased=False).to(torch.float64)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total_count
        )
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
        self.count.copy_(total_count)
