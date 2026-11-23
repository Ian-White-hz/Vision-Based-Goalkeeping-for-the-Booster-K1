"""Camera frame clock: capture at a fixed rate with a random phase, deliver after a lag.

Payloads are flat float tensors so any front end (geometric, rendered, real)
shares the same timing. At most one frame per env is delivered per call: the
newest one that is ready; older ready frames are dropped like a stale queue.
"""

from __future__ import annotations

import torch

from .config import TimingCfg


class FrameClock:
    def __init__(self, num_envs: int, width: int, device, cfg: TimingCfg | None = None):
        self.cfg = cfg or TimingCfg()
        self.n, self.device, k = num_envs, device, self.cfg.queue
        self.period = 1.0 / self.cfg.fps
        self.next_capture = torch.zeros(num_envs, device=device)
        self.latency = torch.zeros(num_envs, device=device)
        self.payload = torch.zeros(num_envs, k, width, device=device)
        self.stamp = torch.zeros(num_envs, k, device=device)
        self.ready_at = torch.full((num_envs, k), float("inf"), device=device)
        self.slot = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rows = torch.arange(num_envs, device=device)

    def reset(self, ids, now: torch.Tensor, randomize: bool = True):
        lo, hi = self.cfg.latency
        m = len(self.rows[ids])
        if randomize:
            self.latency[ids] = torch.empty(m, device=self.device).uniform_(lo, hi)
            phase = torch.rand(m, device=self.device) * self.period
        else:
            self.latency[ids] = 0.5 * (lo + hi)
            phase = torch.zeros(m, device=self.device)
        self.next_capture[ids] = now[ids] + phase
        self.ready_at[ids] = float("inf")

    def due(self, now: torch.Tensor) -> torch.Tensor:
        """Envs that capture a frame at this step (call before push)."""
        return now >= self.next_capture - 1e-6

    def push(self, mask: torch.Tensor, now: torch.Tensor, payload: torch.Tensor):
        if not mask.any():
            return
        rows, slot = self.rows[mask], self.slot[mask]
        jitter = torch.randn(len(rows), device=self.device) * self.cfg.latency_jitter
        self.payload[rows, slot] = payload[mask]
        self.stamp[rows, slot] = now[mask]
        self.ready_at[rows, slot] = now[mask] + (self.latency[mask] + jitter).clamp_min(
            0.0
        )
        self.slot[mask] = (slot + 1) % self.cfg.queue
        # Keep the nominal rate; skip ahead if a step was longer than a period.
        nxt = self.next_capture[mask] + self.period
        self.next_capture[mask] = torch.maximum(nxt, now[mask] + 1e-6)

    def pop(self, now: torch.Tensor):
        """(payload [N,W], stamp [N], delivered [N]) of the newest ready frame."""
        ready = self.ready_at <= now[:, None]
        delivered = ready.any(-1)
        newest = torch.where(ready, self.stamp, -float("inf")).argmax(-1)
        payload = self.payload[self.rows, newest]
        stamp = self.stamp[self.rows, newest]
        self.ready_at[ready] = float("inf")
        return payload, stamp, delivered
