# K1 原项目流程对照分支

分支：`experiment/k1-source-goalkeeper`
任务：`Mjlab-Goalkeeper-Source-Flat-Booster-K1`
基线源码：本地 Humanoid-Goalkeeper，commit `976a81ff19b7306bafbe923d2890066b68a85271`。

目的：把原项目的训练流程独立移到 K1/MuJoCo，检验经过多轮奖励改造的当前任务与原版流程的差异。原 `Mjlab-Goalkeeper-AMP-Flat-Booster-K1` 保留。不是承诺原论文成绩复现。

## 保留的原版行为

- 六区域固定分配给环境；同一环境 reset 后类别不变。支持不能被 6 整除的环境数（原版直接整除会少分配），建议选 6 的倍数。
- 发球起点 x=3..5m、y=-1.8..1.8m、z=.1..1.8m；终点 x=-.6..-.1m。
- 初始 region 的 y/z：0=[.2,1.2]/[.4,1.2]；1 左右镜像；2=[0,1]/[1.2,1.6]；3 镜像；4=[.2,1.2]/[.1,.3]；5 镜像。
- 六类全是弹道球，**最低两类也不是滚地球专用采样**。同一批 reset 共享 .4..1s 飞行时间，play 使用 .5..1s。
- 初始目标是起终点之间 x=.1 的线性插值（不额外修正成无碰撞弹道交点）；球接近 x=.1.. .5 时按当前球位置更新目标。
- 全部 region 的 reach 都针对对应侧的手。没有最近手/脚/躯干选择，没有脚部球路 AABB。
- eereach 的远阶段定位、侧向速度倍率、高区上升速度倍率、直立因子，原样使用。球过身后不套当前任务的 active mask。
- 原奖励权重和 dt 缩放；成功/停球没有额外除以 dt。停球判断采用源代码的 x 速度变化 >2m/s 代理，**不等于严格物理挡球成功率**。
- successland 采用原版 root_z>1 的历史状态和持续奖励（不是一次性落地奖励）。
- 3 秒回合；膝高<.1、重力投影水平模长>.8、双脚平均接触力>1500 终止；没有新增失球惩罚、倒地惩罚、横移引导、heading/tilt 代价、恢复结束终止。
- 每隔超过 500 个环境控制步，在 reset 时根据终止环境平均存活步数/50 更新课程；范围扩张，reach/success/stopball 权重增长、跳跃速度加成增长。
- 十帧历史：K1 单帧75，actor750，critic92；固定观测缩放、球出现/消失和噪声；恢复原 history encoder、6维球估计器、6类 region 分类器。
- 原 HIMPPO、球估计 MSE、分类交叉熵、policy/value smoothness loss、原 rollout minibatch 组织（最后一帧不参与更新）。Actor/Critic 网络 [512,256,256]、ELU。
- 六个原版 AMP 判别器 [512,256]；20个噪声候选、sigma=.3、取最高奖励；共享原 Normalizer、原联合 Adam 和权重衰减。原版跨 reset 两帧窗口行为也保留。
- expert 采样恢复为离散第一帧 + .25..1.25 随机速度比的第二帧插值。原 joint 29→K1 joint 22；使用已有六个 serial retarget 文件。
- `total = .6*task + .4*(.5*AMP)`，本任务不使用上轮的 `amp.reward_scale` 消融开关。
- 默认 rollout100、5 epochs、4 minibatches、6144 env、max_iterations200000（这是公开配置，不是论文实际完成训练预算的声明）。

## 必须明确的 K1 / MuJoCo 差异

1. K1 22 个驱动关节，无腰 DOF。删除 `postwaistdofpos`、`deviation_waist_pitch_joint`，不拿髋关节冒充腰。G1 pelvis/torso 对应 K1 Trunk；手端、膝和脚按 K1 具名 body 映射。
2. 使用 K1 自身的默认站姿、初始根高、增益、关节/速度/力矩限位、惯量与碰撞模型。没有直接复制 G1 的29维 init_pos。控制算法为源项目的逐物理子步显式 PD + torque clipping；动作尺度 .25；起始姿态保持和0..3子步延迟保留。
3. 增益因子 .8..1.2、力矩偏置/注入±1%、随机初始姿态和80% continue_keep；质量/重心/摩擦范围沿用源配置。K1 的绝对增益和力矩上限不能替换成 G1 的硬件参数。
4. PhysX 与 MuJoCo 接触不可逐数等价。恢复系数用线性振子公式映射 MuJoCo 阻尼比，截断到至少 .05，并令 timeconst=.02/zeta 保持刚度不变，属于近似。接触力传感器和求解器也不同。质量随机化保留原质量范围，未声称 PhysX 自动惯量重算与 MuJoCo 的结果相同。
5. MuJoCo 的 termination/reward 读取值存在一个物理子步的派生状态延迟；扰动在 step event 中作用于后续物理步。原版在 post-physics callback 中修改速度，二者顺序不是逐步完全等价。
6. 默认**没有按 K1 身高缩小 G1 的球路和 reward 高度阈值**。高球1.2..1.6m、root_z>1m落地奖励阈值可能超出 K1 的实用范围；不要把无法达到的目标归因于训练预算。若后续缩放，应另做实验并记录，不能称为完全数值一致。
7. checkpoint 保存新格式、Normalizer、优化器和课程状态；恢复的物理 episode/RNG 不承诺原样复现。不接受之前750维观测以前的调参任务 checkpoint。
8. 原版速度变化代理成功与严格 save/goal 指标不同，本任务日志标为 `Source/save_velocity_proxy`，不能和旧任务 safe_save 混作同一指标。

## 文件职责

- `source/vendor/`：固定版本的原 ActorCritic、AMP、HIMPPO、storage、Normalizer 和 reward 方法；保留许可与 source_manifest 哈希。reward 方法仅抽取、替换四元数 API，不改公式。
- `source/settings.py`：region 范围、reward scales。
- `source/state.py`：K1状态映射、原 reward 时序、750/92维观测和终止条件。
- `source/events.py`：reset、弹道、课程、球扰动。
- `source/actions.py`：显式 PD、动作延迟和初始化保持。
- `source/physics.py`：源球质量/半径/惯量、MuJoCo接触适配。
- `source/runner.py`：将原版 HIMPPO 接到 mjlab 的 step、训练、TensorBoard、保存/恢复与 play。
- `source/config.py`、`config/k1/__init__.py`：新任务配置及注册。

## 使用

从零开始，不 resume 以前调参任务。8GB GPU 先用已跑通的192个环境（每region32），其每轮样本仅是原默认的1/32，不能按 iteration 直接对比训练量。

```bash
uv run --no-sync train Mjlab-Goalkeeper-Source-Flat-Booster-K1 \
  --gpu-ids '[0]' --env.scene.num-envs 192 \
  --agent.max-iterations 1000 --agent.run-name source_k1_baseline
```

日志：`logs/rsl_rl/k1_goalkeeper_source/<时间>_source_k1_baseline/`。

```bash
uv run --no-sync play Mjlab-Goalkeeper-Source-Flat-Booster-K1 \
  --checkpoint-file logs/rsl_rl/k1_goalkeeper_source/<运行目录>/model_1000.pt \
  --num-envs 6
```

play 使用六个环境同时显示六类；只有1个环境时会被分配region0。

续训示例（保持并行环境数）：

```bash
uv run --no-sync train Mjlab-Goalkeeper-Source-Flat-Booster-K1 \
  --gpu-ids '[0]' --env.scene.num-envs 192 \
  --agent.resume True --agent.load-run '<运行目录>' \
  --agent.load-checkpoint model_1000.pt --agent.max-iterations 1000 \
  --agent.run-name source_k1_continue
```

本 Runner 的 max_iterations 是本次新增更新数，以上得到 model_2000.pt。

## 验证

- `tests/test_goalkeeper_source_flow.py`：直接解析邻接原项目的 reward 方法，与真实 K1 状态上的所有适用 reward 数值比对；六区域、观测维度、跳跃速度加成；真实 HIMPPO 与六类 AMP 更新，确认 actor/critic/history/估计器/分类器参数都更新；保存/续训及推理一致。
- `tests/test_goalkeeper_source_randomization.py`：随机化 startup/reset、固定 region、课程范围增长、动作延迟、真实更新。
- 正式 GPU `train` 入口：12环境×8步×2轮 smoke；另验证 checkpoint 续训与原版100步rollout。训练成果不由 smoke 判断。
- Ruff（适配层）和 diff whitespace 检查；原版 vendor 保留源码格式。

完整 rollout 曾发现约束容量300不足及恢复系数映射使自碰撞刚度过高；已把容量增至1024并修正 timeconst/zeta 联动。修正后100步随机动作最大关节速度约43.23rad/s、最大单步任务奖励绝对值2.50、最大脚部力分量1391.68N（未训练随机动作探针，非稳定性成绩）。12环境×100步×2轮完整GPU更新的 task/amp 混合分量均值分别约 [-.1387,.1485]、[-.1590,.1649]，均有限。日志：`2026-11-12_22-43-06_source_port_verified`。

本目录源自 Humanoid-Goalkeeper 的改编内容，保留上游 CC BY-NC-SA 4.0 与子目录 BSD 声明；见 `source/vendor/NOTICE` 和 LICENSE 文件。

最终验证：12环境原版完整配置续训到第3轮通过；192环境×100步×5epochs×4minibatches 在本机 RTX3070Ti 8GB 完成一轮，task/amp均值约-.1440/.1472。play 关闭原版所关闭的观测噪声、20%随机初始角分支及机器人推力，保留原来仍开启的摩擦/球扰动等。
