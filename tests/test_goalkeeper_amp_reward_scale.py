"""AMP-only ablation: identical rollout, real PPO update, old checkpoint resume."""
from pathlib import Path
from tempfile import TemporaryDirectory

import mjlab  # noqa: F401 -- initialize registered tasks before importing runners
import torch

from goalkeeper_amp_test_utils import make_runner
from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards

kwargs = dict(style_weight=0.4, step_dt=0.02, scale_rewards_by_dt=True)
task = torch.tensor([2.0, -3.0])
amp = torch.tensor([0.8, 0.4])
base = mix_amp_rewards(task, amp, **kwargs)
for scale in (0.0, 0.25, 1.0):
    total, task_part, style_part = mix_amp_rewards(task, amp, reward_scale=scale, **kwargs)
    torch.testing.assert_close(task_part, base[1])
    torch.testing.assert_close(style_part, base[2] * scale)
    torch.testing.assert_close(total, task_part + style_part)
for scale in (-1.0, float('nan'), float('inf')):
    try:
        mix_amp_rewards(task, amp, reward_scale=scale, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid reward_scale accepted')

with TemporaryDirectory() as tmp:
    source, _ = make_runner()
    source.learn(1)
    path = Path(tmp) / 'old.pt'
    source.save(str(path))
    payload = torch.load(path, weights_only=True)
    del payload['infos']['categorical_amp_runner']['reward_scale']
    torch.save(payload, path)
    summaries = []
    for scale in (1.0, 0.25):
        runner, _ = make_runner(reward_scale=scale)
        runner.load(str(path))
        assert runner.current_learning_iteration == 1
        assert runner.amp_cfg['reward_scale'] == scale
        torch.manual_seed(123)
        runner.learn(1)
        summaries.append(runner._last_iteration_summary['reward_means'])
        assert runner.current_learning_iteration == 2
        saved = Path(tmp) / f'scale_{scale}.pt'
        runner.save(str(saved))
        state = torch.load(saved, weights_only=True)['infos']['categorical_amp_runner']
        assert state['reward_scale'] == scale
    for name in ('amp_raw', 'task_component'):
        torch.testing.assert_close(torch.tensor(summaries[0][name]), torch.tensor(summaries[1][name]))
    torch.testing.assert_close(
        torch.tensor(summaries[1]['style_component']),
        torch.tensor(summaries[0]['style_component'] * 0.25),
    )
print('AMP倍率：任务系数不变、旧checkpoint续训、真实PPO更新、保存元数据均通过')
