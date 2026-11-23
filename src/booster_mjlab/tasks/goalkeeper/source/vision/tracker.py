"""Batched ballistic Kalman filter for the ball; identical in simulation and on the robot.

State [p, v] in a gravity-aligned tracking frame (z up). Measurements are 3D
points with a covariance and a capture lag: the filter stays at the current
time and relates a lagged measurement through the exact backward ballistic
map, p(t - L) = p(t) - L v(t) + g L^2 / 2 (g pointing down).
"""

from __future__ import annotations

import torch

from .config import TrackerCfg


class BallTracker:
    def __init__(self, num_envs: int, device, cfg: TrackerCfg | None = None):
        self.cfg = cfg or TrackerCfg()
        self.n, self.device = num_envs, device
        self.g = torch.tensor([0.0, 0.0, -self.cfg.gravity], device=device)
        self.x = torch.zeros(num_envs, 6, device=device)
        self.P = torch.eye(6, device=device).repeat(num_envs, 1, 1)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hits = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rejects = torch.zeros_like(self.hits)
        self.since_update = torch.full((num_envs,), 1e3, device=device)
        self.eye3 = torch.eye(3, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.x[ids] = 0.0
        self.P[ids] = torch.eye(6, device=self.device)
        self.initialized[ids] = False
        self.hits[ids] = 0
        self.rejects[ids] = 0
        self.since_update[ids] = 1e3

    def predict(self, dt: float):
        c = self.cfg
        p, v = self.x[:, :3], self.x[:, 3:]
        p = p + v * dt + 0.5 * self.g * dt * dt
        v = v + self.g * dt
        if c.ground_z is not None:
            bounce = (
                (p[:, 2] < c.ground_z + c.ball_radius)
                & (v[:, 2] < 0)
                & self.initialized
            )
            v = v.clone()
            v[bounce, 2] = -c.restitution * v[bounce, 2]
        self.x = torch.cat((p, v), -1)
        F = torch.eye(6, device=self.device)
        F[:3, 3:] = dt * self.eye3
        q = c.accel_noise**2
        Q = torch.zeros(6, 6, device=self.device)
        Q[:3, :3] = q * dt**3 / 3 * self.eye3
        Q[:3, 3:] = Q[3:, :3] = q * dt**2 / 2 * self.eye3
        Q[3:, 3:] = q * dt * self.eye3
        self.P = F @ self.P @ F.T + Q
        self.since_update += dt

    def update(
        self, z: torch.Tensor, R: torch.Tensor, lag: torch.Tensor, mask: torch.Tensor
    ):
        """z [N,3] measured position at time now - lag, R [N,3,3], mask [N] bool."""
        c = self.cfg
        lag = lag.clamp_min(0.0)
        fresh = mask & ~self.initialized
        tracked = mask & self.initialized

        # Innovation through the lagged measurement model H = [I, -L I].
        H = torch.zeros(self.n, 3, 6, device=self.device)
        H[:, :, :3] = self.eye3
        H[:, :, 3:] = -lag[:, None, None] * self.eye3
        zhat = (
            self.x[:, :3]
            - lag[:, None] * self.x[:, 3:]
            + 0.5 * self.g * lag[:, None] ** 2
        )
        y = z - zhat
        S = H @ self.P @ H.transpose(1, 2) + R
        Sinv = torch.linalg.inv(S)
        d2 = torch.einsum("ni,nij,nj->n", y, Sinv, y)
        accept = tracked & (d2 < c.gate)
        K = self.P @ H.transpose(1, 2) @ Sinv
        x_new = self.x + (K @ y[:, :, None])[:, :, 0]
        P_new = (torch.eye(6, device=self.device) - K @ H) @ self.P
        self.x = torch.where(accept[:, None], x_new, self.x)
        self.P = torch.where(
            accept[:, None, None], 0.5 * (P_new + P_new.transpose(1, 2)), self.P
        )

        rejected = tracked & ~accept
        self.rejects = torch.where(
            rejected, self.rejects + 1, torch.where(accept, 0, self.rejects)
        )
        restart = fresh | (rejected & (self.rejects >= c.reinit_after))
        if restart.any():
            # Position from the measurement (lag ignored: velocity unknown yet).
            sv2 = c.init_vel_std**2
            P0 = torch.zeros(self.n, 6, 6, device=self.device)
            P0[:, :3, :3] = R + (lag**2 * sv2)[:, None, None] * self.eye3
            P0[:, 3:, 3:] = sv2 * self.eye3
            x0 = torch.cat((z, torch.zeros_like(z)), -1)
            self.x = torch.where(restart[:, None], x0, self.x)
            self.P = torch.where(restart[:, None, None], P0, self.P)
            self.initialized |= restart
            self.hits = torch.where(restart, 1, self.hits)
            self.rejects = torch.where(restart, 0, self.rejects)
        self.hits = torch.where(accept, self.hits + 1, self.hits)
        self.since_update = torch.where(accept | restart, lag, self.since_update)

    @property
    def position(self) -> torch.Tensor:
        return self.x[:, :3]

    @property
    def velocity(self) -> torch.Tensor:
        return self.x[:, 3:]

    @property
    def valid(self) -> torch.Tensor:
        c = self.cfg
        pos_std = self.P[:, [0, 1, 2], [0, 1, 2]].sum(-1).sqrt()
        return (
            self.initialized
            & (self.hits >= c.confirm)
            & (self.since_update <= c.max_coast)
            & (pos_std <= c.max_sigma)
        )
