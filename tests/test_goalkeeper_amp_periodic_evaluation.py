"""Actual PPO updates: evaluation isolation, scheduling, best selection and resume."""

import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mjlab  # noqa: F401
import numpy as np
import torch
from mjlab.utils.os import get_checkpoint_path

from goalkeeper_amp_test_utils import make_runner as make_test_runner
from booster_mjlab.amp.runners.periodic_evaluation import preserve_rng


def make_runner(**kwargs):
    return make_test_runner(obs_normalization=True, **kwargs)


class FakeEvaluator:
    signature = "fixed-test-suite-v1"

    def __init__(self, **kwargs):
        random.random(), np.random.rand(), torch.rand(1)
        self.calls = 0

    def evaluate(self, actor):
        assert not actor.training
        self.calls += 1
        # Deliberately disturb all global generators and the copied actor.
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        os.environ["PYTHONHASHSEED"] = "999"
        for p in actor.parameters():
            p.zero_()
        score = [.2, .3, .1, .3][self.calls - 1]
        return {"selection_score": [score], **{
            name: {"metrics": {"macro": {"safe_save_rate": score}}}
            for name in ("validation", "holdout")}}


def enable(runner):
    runner.periodic_evaluation.cfg.update(
        interval=2, evaluator_factory="unused:factory", evaluate_at_start=True,
        evaluate_at_end=True,
    )


with TemporaryDirectory(prefix="goalkeeper_periodic_eval_") as temp:
    torch.manual_seed(47)
    baseline, _ = make_runner(log_dir=str(Path(temp) / "baseline"))
    baseline.learn(5)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(47)
    runner, _ = make_runner(log_dir=str(Path(temp) / "evaluated"))
    enable(runner)
    with patch("booster_mjlab.amp.runners.periodic_evaluation.resolve_amp_callable",
               return_value=FakeEvaluator):
        runner.learn(5)
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
    for actual, expected in ((runner.alg.actor.state_dict(), baseline.alg.actor.state_dict()),
                             (runner.alg.critic.state_dict(), baseline.alg.critic.state_dict()),
                             (runner.amp_model.state_dict(), baseline.amp_model.state_dict())):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(runner.env.q, baseline.env.q, rtol=0, atol=0)
    assert runner.env.common_step_counter == baseline.env.common_step_counter == 20
    assert runner.total_timesteps == baseline.total_timesteps == 40
    directory = Path(runner.logger.log_dir)
    reports = [json.loads(p.read_text()) for p in sorted((directory / "evaluation").glob("*.json"))]
    assert [r["iteration"] for r in reports] == [0, 2, 4, 5]
    assert [r["selected_best"] for r in reports] == [True, True, False, False]
    best = torch.load(directory / "model_best.pt", weights_only=True)
    assert best["iter"] == best["infos"]["periodic_evaluation"]["best"]["iteration"] == 2
    assert torch.load(directory / "model_5.pt", weights_only=True)["iter"] == 5
    assert runner.logger.writer is None
    assert get_checkpoint_path(directory.parent, directory.name,
                               runner.cfg["load_checkpoint"]).name == "model_5.pt"

    restored, _ = make_runner(log_dir=str(directory))
    restored.load(str(directory / "model_5.pt"))
    enable(restored)
    with patch("booster_mjlab.amp.runners.periodic_evaluation.resolve_amp_callable",
               return_value=FakeEvaluator):
        restored._evaluate_policy(force=True)
    assert restored.periodic_evaluation.evaluator.calls == 0  # no duplicate iteration 5
    assert restored.periodic_evaluation.state["best"]["iteration"] == 2

    # Loading an older numbered snapshot must respect a newer best artifact.
    stale = torch.load(directory / "model_5.pt", weights_only=True)
    stale["infos"]["periodic_evaluation"]["best"]["iteration"] = 0
    stale["infos"]["periodic_evaluation"]["best"]["score"] = [.2]
    torch.save(stale, directory / "model_1.pt")
    older, _ = make_runner(log_dir=str(directory))
    older.load(str(directory / "model_1.pt"))
    assert older.periodic_evaluation.state["best"]["iteration"] == 2

    fresh, _ = make_runner(log_dir=str(Path(temp) / "new-output"))
    fresh.load(str(directory / "model_5.pt"))
    assert fresh.periodic_evaluation.state == {}  # new directory needs local best
    warm, _ = make_runner(warm_start=True, log_dir=str(directory))
    warm.load(str(directory / "model_5.pt"))
    assert warm.periodic_evaluation.state == {} and warm.current_learning_iteration == 0

    # A changed evaluation protocol starts a fresh comparison, even on resume.
    restored.periodic_evaluation.evaluator.signature = "changed-suite"
    restored._evaluate_policy(force=True)
    assert restored.periodic_evaluation.state["best"]["iteration"] == 5
    previous = (directory / "model_best.pt").read_bytes()
    restored.current_learning_iteration = 6
    restored.periodic_evaluation.evaluator = FakeEvaluator()
    restored.periodic_evaluation.evaluator.signature = "third-suite"
    old_state = dict(restored.periodic_evaluation.state)
    with patch.object(restored, "save", side_effect=OSError("disk-full")):
        try:
            restored.periodic_evaluation.run(restored, force=True)
        except OSError:
            pass
        else:
            raise AssertionError("save error swallowed")
    assert (directory / "model_best.pt").read_bytes() == previous
    assert restored.periodic_evaluation.state == {}  # incompatible old protocol cleared
    assert old_state["signature"] == "changed-suite"

    # Evaluation failure preserves the completed training checkpoint and closes logging.
    broken, _ = make_runner(log_dir=str(Path(temp) / "broken"))
    enable(broken)
    with patch("booster_mjlab.amp.runners.periodic_evaluation.resolve_amp_callable",
               side_effect=ValueError("broken configuration")):
        try:
            broken.learn(1)
        except ValueError:
            pass
        else:
            raise AssertionError("evaluation failure swallowed")
    assert (Path(broken.logger.log_dir) / "model_0.pt").is_file()
    assert broken.logger.writer is None

random.seed(41)
np.random.seed(41)
torch.manual_seed(41)
expected = random.random(), float(np.random.rand()), torch.rand(2)
random.seed(41)
np.random.seed(41)
torch.manual_seed(41)
hash_before = os.environ.get("PYTHONHASHSEED")
try:
    with preserve_rng():
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        os.environ["PYTHONHASHSEED"] = "1"
        raise RuntimeError("probe")
except RuntimeError:
    pass
assert random.random() == expected[0] and float(np.random.rand()) == expected[1]
torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
assert os.environ.get("PYTHONHASHSEED") == hash_before
print("Evaluation scheduling, exact PPO/AMP/RNG isolation, outcome selection, resume/warm-start, protocol changes and failure protection passed")
