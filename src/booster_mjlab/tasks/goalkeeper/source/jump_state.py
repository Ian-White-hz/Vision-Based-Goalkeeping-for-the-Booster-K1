"""K1 takeoff detection; retains upstream repeated landing reward semantics."""

import torch


class JumpState:
    def __init__(self, num_envs, device):
        self.initial_height = torch.zeros(num_envs, device=device)
        self.supported = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.air_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.jumped = torch.zeros_like(self.supported)

    def reset(self, ids, height):
        self.initial_height[ids] = height
        self.supported[ids] = False
        self.air_steps[ids] = 0
        self.jumped[ids] = False

    def update(self, height, foot_force_z):
        contact = foot_force_z > 1.0
        self.supported |= contact.all(-1)
        air = ~contact.any(-1)
        self.air_steps = torch.where(air, self.air_steps + 1, 0)
        self.jumped |= (
            self.supported
            & (self.air_steps >= 2)
            & (height - self.initial_height >= 0.05)
        )
        return self.jumped
