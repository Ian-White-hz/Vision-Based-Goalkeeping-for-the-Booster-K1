# AMP 强度对照实验

目的：检验 AMP 是否抑制远侧来球时的横移。不能仅由 AMP 回报较大推断因果，也不能预先断言参考动作没有横移。

增加 `agent.amp.reward_scale`（默认 1，有限非负）。

- 对照：`0.6 * task_reward + 0.2 * AMP_score`。
- 实验：`0.6 * task_reward + 0.05 * AMP_score`（scale=0.25）。
- 不修改任务奖励、发球、观测、动作、判别器损失或更新频率。
- scale=0 仍训练判别器，只取消其对策略回报的贡献。
- checkpoint 保存倍率，但恢复时采用本次配置的倍率，以便从同一 checkpoint 分叉；旧 checkpoint 无需迁移。
- 后续再次 resume 实验组必须继续显式指定 0.25，默认值仍为 1。
- TensorBoard 新增 `AMP/reward_scale`；原 `AMP/step_reward/style_component` 是实际加权贡献。
- 冻结诊断脚本从 checkpoint 读取倍率（旧文件按 1），并写入 results.json。

## 两组训练

依次执行，避免同时占用 8GB 显存。两组均完整恢复同一个 model_4000，保留 PPO、判别器与优化器；不是从零训练，也不重置判别器。每组再训练 1000 次，最终 model_5000。

```bash
uv run --no-sync train Mjlab-Goalkeeper-AMP-Flat-Booster-K1 \
  --gpu-ids '[0]' --env.scene.num-envs 512 --agent.seed 123 \
  --agent.resume True --agent.warm-start False \
  --agent.load-run 2026-11-12_16-36-55_amp_test_side_movement_v2 \
  --agent.load-checkpoint model_4000.pt \
  --agent.max-iterations 1000 --agent.amp.reward-scale 1.0 \
  --agent.run-name amp_scale_control

uv run --no-sync train Mjlab-Goalkeeper-AMP-Flat-Booster-K1 \
  --gpu-ids '[0]' --env.scene.num-envs 512 --agent.seed 123 \
  --agent.resume True --agent.warm-start False \
  --agent.load-run 2026-11-12_16-36-55_amp_test_side_movement_v2 \
  --agent.load-checkpoint model_4000.pt \
  --agent.max-iterations 1000 --agent.amp.reward-scale 0.25 \
  --agent.run-name amp_scale_quarter
```

相同种子只控制随机条件，不保证 GPU 逐位确定性；恢复也不恢复历史物理状态。两组都从新回合开始。

## 判据

比较相同来球种子的横向根部位移、双脚位移、来球到达前足部覆盖误差、各类别 safe_save/goal/fall。不要用总 reward 高低判断，因为 AMP 尺度已不同。

若实验组横移、覆盖与成功率提高且倒地没有明显恶化，支持 AMP 约束过强的假设；若只更易倒或仍不移动，不能认定 AMP 是主因。单种子短程结果只用于筛查；不能保证 1000 次足以摆脱旧策略。

## 实现和验证

倍率沿配置 → Runner → iteration → step → reward_mixing 传递；元数据和诊断读取同步更新。

新增 `test_goalkeeper_amp_reward_scale.py`：任务项不变、AMP 缩放比例、非法输入、旧 checkpoint 恢复、同种子真实 PPO 更新和保存。另运行奖励混合、Runner checkpoint/TensorBoard 回归，以及真实物理环境、真实 expert 的短 PPO/AMP 更新测试。

本记录仅说明测试支持与短程验证，不宣称两组长期训练已完成。
