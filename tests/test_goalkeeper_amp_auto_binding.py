# 与正式训练入口保持一致，先初始化 mjlab。
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mjlab  # noqa: F401
import torch

import goalkeeper_amp_test_utils as test_utils
from goalkeeper_amp_test_utils import (
    FakeExpertSampler,
    make_runner,
    read_motion_ids,
)


runner, _ = make_runner(bind_interfaces=False)

runner.amp_cfg["dataset_root"] = "fake_dataset_for_test"
runner.amp_cfg["expert_sampler_factory"] = (
    "goalkeeper_amp_test_utils:create_fake_expert_sampler"
)
runner.amp_cfg["motion_id_reader"] = "goalkeeper_amp_test_utils:read_motion_ids"

# 创建 Runner 后还没有绑定，更没有读取参考动作。
assert runner.amp_updater.expert_sampler is None
assert runner._read_motion_ids is None

runner.learn(1)

sampler = runner.amp_updater.expert_sampler

assert isinstance(sampler, FakeExpertSampler)
assert sampler.dataset_root == "fake_dataset_for_test"
assert sampler.device == torch.device("cpu")
assert runner._read_motion_ids is read_motion_ids
assert runner.current_learning_iteration == 1
assert sampler.requests

print("配置解析、延迟创建和自动绑定检查通过")


# 再次训练，应继续使用同一个采样器。
request_count = len(sampler.requests)

runner.learn(1)

assert runner.amp_updater.expert_sampler is sampler
assert len(sampler.requests) > request_count
assert runner.current_learning_iteration == 2

print("重复 learn 不重新创建 sampler 检查通过")


# 显式手动绑定优先，即使配置路径不可用也不覆盖。
manual_runner, manual_sampler = make_runner()

manual_runner.amp_cfg["expert_sampler_factory"] = (
    "nonexistent_goalkeeper_module:factory"
)
manual_runner.amp_cfg["motion_id_reader"] = "nonexistent_goalkeeper_module:reader"

manual_runner.learn(1)

assert manual_runner.amp_updater.expert_sampler is manual_sampler
assert manual_runner._read_motion_ids is read_motion_ids

print("手动绑定优先检查通过")


factory_path = "goalkeeper_amp_test_utils:create_fake_expert_sampler"
reader_path = "goalkeeper_amp_test_utils:read_motion_ids"


def assert_setup_failure(broken_runner, expected_error):
    """Configuration failures must not step physics or leave a partial binding."""
    try:
        broken_runner.learn(1)
    except expected_error:
        pass
    else:
        raise AssertionError("Invalid AMP configuration was accepted")
    assert broken_runner.env.step_count == 0
    assert broken_runner.current_learning_iteration == 0
    assert broken_runner.alg.storage.step == 0
    assert broken_runner.logger.writer is None
    assert broken_runner.amp_updater.expert_sampler is None
    assert broken_runner._read_motion_ids is None


for factory_entry, reader_entry, error_type in (
    (None, None, RuntimeError),
    (factory_path, None, RuntimeError),
    (None, reader_path, RuntimeError),
    ("math:pi", reader_path, TypeError),
):
    broken, _ = make_runner(
        bind_interfaces=False,
        expert_sampler_factory=factory_entry,
        motion_id_reader=reader_entry,
    )
    assert_setup_failure(broken, error_type)

with patch.object(
    test_utils,
    "create_fake_expert_sampler",
    wraps=test_utils.create_fake_expert_sampler,
) as factory_spy:
    bad_reader, _ = make_runner(
        bind_interfaces=False,
        expert_sampler_factory=factory_path,
        motion_id_reader="goalkeeper_amp_test_utils:no_such_reader",
    )
    factory_spy.assert_not_called()
    assert_setup_failure(bad_reader, AttributeError)
    factory_spy.assert_not_called()

print("错误配置提前拒绝、reader 错误不创建 sampler 检查通过")


class WrongOrderSampler(FakeExpertSampler):
    joint_names = tuple(reversed(FakeExpertSampler.joint_names))


for factory_result, error_type in (
    (object(), TypeError),
    (SimpleNamespace(sample=lambda *args, **kwargs: None), ValueError),
    (WrongOrderSampler(), ValueError),
):
    with patch.object(
        test_utils,
        "create_fake_expert_sampler",
        return_value=factory_result,
    ):
        broken, _ = make_runner(
            bind_interfaces=False,
            expert_sampler_factory=factory_path,
            motion_id_reader=reader_path,
        )
        assert_setup_failure(broken, error_type)

print("factory 返回非法 sampler 时不留下半绑定状态，检查通过")


with TemporaryDirectory(prefix="goalkeeper_amp_binding_") as directory:
    checkpoint_path = Path(directory) / "model.pt"
    runner.save(str(checkpoint_path))

    # Invalid entries configured BEFORE construction must not prevent play.
    play_runner, _ = make_runner(
        bind_interfaces=False,
        expert_sampler_factory="nonexistent_goalkeeper_module:factory",
        motion_id_reader="nonexistent_goalkeeper_module:reader",
    )
    with patch.object(
        play_runner,
        "_ensure_amp_interfaces",
        side_effect=AssertionError("Inference must not initialize expert data"),
    ):
        play_runner.load(str(checkpoint_path), load_cfg={"actor": True})
        policy = play_runner.get_inference_policy(device="cpu")
        with torch.no_grad():
            actions = policy(play_runner.env.get_observations())
        assert actions.shape == (play_runner.env.num_envs, play_runner.env.num_actions)
        assert torch.isfinite(actions).all().item()
        play_runner.env.step(actions)
    assert play_runner.amp_updater.expert_sampler is None
    assert play_runner._read_motion_ids is None
    assert play_runner.current_learning_iteration == 0
    print("Actor-only 加载、推理与模拟环境 step 不读取 expert，检查通过")

    # Full resume restores learning state without eagerly constructing a sampler.
    with patch.object(
        test_utils,
        "create_fake_expert_sampler",
        wraps=test_utils.create_fake_expert_sampler,
    ) as factory_spy:
        resumed, _ = make_runner(
            bind_interfaces=False,
            expert_sampler_factory=factory_path,
            motion_id_reader=reader_path,
        )
        resumed.load(str(checkpoint_path))
        factory_spy.assert_not_called()
        assert resumed.current_learning_iteration == 2
        assert resumed.amp_updater.expert_sampler is None
        resumed.learn(1)
        factory_spy.assert_called_once()
        assert resumed.current_learning_iteration == 3
        bound_sampler = resumed.amp_updater.expert_sampler
        resumed.learn(1)
        factory_spy.assert_called_once()
        assert resumed.amp_updater.expert_sampler is bound_sampler
        assert resumed.current_learning_iteration == 4
    print("resume 延迟自动绑定与重复 learn 复用 sampler 检查通过")
