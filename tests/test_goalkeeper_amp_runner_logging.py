"""Real TensorBoard files and Runner checkpoints; fake physics, real PPO."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from goalkeeper_amp_test_utils import FakeEnv, make_runner


class LoggingFakeEnv(FakeEnv):
    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        extras["log"] = {"Task/test_metric": torch.tensor([0.25])}
        return obs, rewards, dones, extras


def read_events(directory):
    return EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()


torch.manual_seed(42)
with TemporaryDirectory(prefix="goalkeeper_amp_runner_") as directory:
    log_dir = Path(directory) / "training"
    runner, _ = make_runner(env=LoggingFakeEnv(), log_dir=str(log_dir), save_interval=2)
    with patch.object(runner, "save", wraps=runner.save) as save_spy:
        runner.learn(3)
        assert [Path(call.args[0]).name for call in save_spy.call_args_list] == [
            "model_2.pt",
            "model_3.pt",
        ]
    assert runner.current_learning_iteration == 3
    assert runner.total_timesteps == runner.logger.tot_timesteps == 24
    assert runner.total_time > 0.0
    assert runner.logger.writer is None
    assert not list(log_dir.glob("*.tmp"))
    events = read_events(log_dir)
    required = {
        "Loss/value",
        "Loss/surrogate",
        "Loss/entropy",
        "Loss/amp",
        "Loss/learning_rate",
        "Policy/mean_std",
        "Perf/collection_time",
        "Perf/learning_time",
        "Perf/total_timesteps",
        "Train/mean_reward",
        "Train/completed_iterations",
        "AMP/step_reward/amp_raw",
        "AMP/step_reward/task_component",
        "AMP/step_reward/style_component",
        "AMP/step_reward/total",
        "AMP/valid_fraction",
        "AMP/reward_scale",
        "AMP/normalizer_count",
        "Discriminator/lefthand/total_loss",
        "Discriminator/rightstep/gradient_penalty",
        "Task/test_metric",
    }
    assert required <= set(events.Tags()["scalars"])
    for tag in required:
        values = events.Scalars(tag)
        assert [event.step for event in values] == [0, 1, 2], tag
        assert all(math.isfinite(event.value) for event in values), tag
    assert [e.value for e in events.Scalars("AMP/valid_fraction")] == [0.875, 1.0, 1.0]
    assert [e.value for e in events.Scalars("Perf/total_timesteps")] == [
        8.0,
        16.0,
        24.0,
    ]
    assert [e.value for e in events.Scalars("Train/completed_iterations")] == [
        1.0,
        2.0,
        3.0,
    ]
    print("真实 TensorBoard 指标、环境 extras、periodic/final 保存与关闭检查通过")

    checkpoint = log_dir / "model_3.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["iter"] == 3
    assert payload["infos"]["env_state"]["common_step_counter"] == 12
    metadata = payload["infos"]["categorical_amp_runner"]
    assert metadata["total_timesteps"] == 24
    assert metadata["iteration_semantics"] == "completed_updates"
    assert payload["categorical_amp"]["schema_version"] == 2

    restored, _ = make_runner(log_dir=str(log_dir), save_interval=2)
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 3
    assert restored.total_timesteps == restored.logger.tot_timesteps == 24
    assert restored.total_time == runner.total_time
    assert restored.env.common_step_counter == 12
    for name, value in runner.amp_model.state_dict().items():
        torch.testing.assert_close(
            restored.amp_model.state_dict()[name], value, rtol=0, atol=0
        )
    assert len(restored.alg.optimizer.state) == len(runner.alg.optimizer.state)
    # Physical poses are intentionally fresh; History must match the new env.
    torch.testing.assert_close(
        restored.amp_collector.history._previous_q, restored.env.q, rtol=0, atol=0
    )
    with patch.object(restored, "save", wraps=restored.save) as save_spy:
        restored.learn(1)
        assert [Path(call.args[0]).name for call in save_spy.call_args_list] == [
            "model_4.pt"
        ]
    assert restored.current_learning_iteration == 4
    assert restored.total_timesteps == 32
    assert restored.env.common_step_counter == 16
    events = read_events(log_dir)
    assert [e.step for e in events.Scalars("Loss/amp")] == [0, 1, 2, 3]
    assert [e.value for e in events.Scalars("Perf/total_timesteps")] == [8, 16, 24, 32]
    assert restored.logger.writer is None
    print("Runner resume：计数器、共享统计、联合 Adam、History 重建及日志续接检查通过")

    # A write failure must not corrupt the previous checkpoint or leave a temp.
    with patch("torch.save", side_effect=OSError("simulated disk error")):
        try:
            runner.save(str(checkpoint))
        except OSError:
            pass
        else:
            raise AssertionError("Checkpoint write failure was swallowed")
    assert torch.load(checkpoint, weights_only=True)["iter"] == 3
    assert not list(log_dir.glob("*.tmp"))
    print("checkpoint 原子写入失败保护检查通过")

    # Warm start supports the existing ordinary PPO baseline, without AMP state.
    baseline = Path(directory) / "ppo_baseline.pt"
    baseline_state = {k: v for k, v in payload.items() if k != "categorical_amp"}
    baseline_state["infos"] = {"env_state": {"common_step_counter": 999}}
    torch.save(baseline_state, baseline)
    warm, _ = make_runner(warm_start=True, style_weight=0.7)
    warm.env.common_step_counter = 7
    fresh_amp = {
        name: value.clone() for name, value in warm.amp_model.state_dict().items()
    }
    warm.load(str(baseline))
    assert warm.current_learning_iteration == warm.total_timesteps == 0
    assert warm.env.common_step_counter == 7
    assert not warm.alg.optimizer.state
    for name, value in warm.amp_model.state_dict().items():
        torch.testing.assert_close(value, fresh_amp[name], rtol=0, atol=0)
    for name, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(
            warm.alg.actor.state_dict()[name], value, rtol=0, atol=0
        )
    warm.learn(1)
    assert warm.current_learning_iteration == 1
    assert warm.total_timesteps == warm.logger.tot_timesteps == 8
    assert warm.logger.writer is None
    try:
        warm.load(str(baseline))
    except RuntimeError as error:
        assert "fresh runner" in str(error)
    else:
        raise AssertionError("Warm start silently reused an old optimizer")

    actor_only, _ = make_runner()
    actor_only.env.common_step_counter = 8
    actor_only.load(str(checkpoint), load_cfg={"actor": True})
    assert actor_only.env.common_step_counter == 8
    assert actor_only.current_learning_iteration == actor_only.total_timesteps == 0
    print("普通 PPO warm start、全新 D/优化器和 actor-only 不恢复环境计数检查通过")

    incompatible, _ = make_runner(style_weight=0.1)
    try:
        incompatible.load(str(checkpoint))
    except ValueError as error:
        assert "style_weight" in str(error)
    else:
        raise AssertionError("Changed reward semantics silently resumed")
    try:
        incompatible.load(str(baseline))
    except ValueError:
        pass
    else:
        raise AssertionError("PPO-only checkpoint silently resumed as full AMP")

    failed_dir = Path(directory) / "failed"
    failed, _ = make_runner(log_dir=str(failed_dir))
    with patch.object(
        failed.alg, "update", side_effect=RuntimeError("simulated update error")
    ):
        try:
            failed.learn(1)
        except RuntimeError as error:
            assert "simulated update error" in str(error)
        else:
            raise AssertionError("Update failure was swallowed")
    assert failed.logger.writer is None
    assert failed.current_learning_iteration == 0
    assert not list(failed_dir.glob("model_*.pt"))
    try:
        failed.save(str(failed_dir / "partial.pt"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete rollout was saved as resumable state")
    invalid_interval, _ = make_runner(save_interval=0)
    try:
        invalid_interval.learn(1)
    except ValueError as error:
        assert "save_interval" in str(error)
    else:
        raise AssertionError("Invalid save interval accepted")
    print("不兼容恢复拒绝、异常关闭 writer 和不保存半轮状态检查通过")
