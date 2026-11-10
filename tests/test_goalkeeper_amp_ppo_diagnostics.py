"""Diagnostics describe the real gradients before joint clipping, including fixed LR."""
import math
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from goalkeeper_amp_test_utils import make_runner

torch.manual_seed(42)
with TemporaryDirectory() as directory:
    runner, _ = make_runner(log_dir=directory, epochs=2, minibatches=2)
    observed = []
    original = torch.nn.utils.clip_grad_norm_

    def record(parameters, *args, **kwargs):
        parameters = list(parameters)
        actor = torch.cat([p.grad.flatten() for p in runner.alg.actor.parameters()
                           if p.grad is not None]).norm().item()
        critic = torch.cat([p.grad.flatten() for p in runner.alg.critic.parameters()
                            if p.grad is not None]).norm().item()
        observed.append((actor, critic))
        return original(parameters, *args, **kwargs)

    lr = runner.alg.learning_rate
    with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=record):
        runner.learn(1)
    diagnostics = runner.alg.last_ppo_diagnostics
    assert len(observed) == 4
    for index, name in enumerate(("actor", "critic")):
        assert math.isclose(diagnostics[f"{name}_grad_norm_pre_clip"],
                            sum(row[index] for row in observed) / 4, rel_tol=1e-5)
    assert runner.alg.learning_rate == lr
    assert 0 <= diagnostics["clip_fraction"] <= 1
    assert 0 < diagnostics["grad_clip_scale"] <= 1
    assert abs(diagnostics["kl_first"]) < 1e-5
    events = EventAccumulator(directory).Reload()
    for name, value in diagnostics.items():
        assert math.isfinite(value)
        assert math.isclose(events.Scalars(f"PPO/{name}")[0].value, value,
                            rel_tol=1e-5, abs_tol=1e-7)

print("Fixed-LR KL, clipping fraction, pre-clip gradients and TensorBoard verified")
