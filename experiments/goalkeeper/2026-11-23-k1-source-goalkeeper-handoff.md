# K1 守门员交接文档（2026-11-23）

分支：`experiment/k1-source-goalkeeper`
任务：`Mjlab-Goalkeeper-Source-Flat-Booster-K1`，是上游
[Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) 的 mjlab 移植版，上游参考 commit 976a81f。

## 一句话现状

- **policy**：中路和低球已经能挡住相当一部分，高球基本挡不住，总挡出率稳定在 .41–.47。
- **视觉**：相机球输入在仿真里已经跑通，还没接真机数据。

当前在做的事情：**查明高球挡不住的原因并修复。**

## 1. 阶段进度

| 模块 | 状态 |
|---|---|
| 源任务移植（奖励、AMP、课程、6 个射门区域） | 完成，奖励项和权重与上游逐项对齐，有测试保证 |
| policy 训练 | 有可用的 checkpoint（见第 2 节）。中路、低球可用，高球不行 |
| 视觉球输入（仿真） | 完成：synthetic / rendered / YOLO 三种模式、跟踪器、YOLO 数据集和训练脚本 |
| 视觉微调 policy | 做过一次，加视觉噪声后挡出率从约 .43 降到约 .38 |
| 真机 | 未开始：没有真机相机数据和标定，YOLO 只用仿真图片训练过 |

## 2. 可用的 checkpoint（已提交到仓库）

checkpoint 都在 `logs/rsl_rl/k1_goalkeeper_source/<run>/` 下，每个 run 目录里都带 `params/`（训练配置）。

评估方式：真实球位置输入，1200 脚球，每个区域 200 脚。每格是"手臂碰到球的比例 / 球被挡在门外的比例"。

| run / checkpoint | 总挡出率 | mid-L | mid-R | high-L | high-R | low-L | low-R |
|---|---|---|---|---|---|---|---|
| `2026-11-17_11-28-20_source_k1_upstream_like_cont/model_12000.pt`（**基线，后续训练都从这里开始**） | .444 | .85/.52 | .96/.59 | .45/.05 | .24/.02 | .88/.68 | .95/.79 |
| `2026-11-22_16-35-56_source_k1_highreach/model_15000.pt` | .468 | .85/.62 | .82/.55 | .38/.02 | .04/.00 | .92/.94 | .93/.68 |
| `2026-11-17_21-48-09_source_k1_20k/model_12400.pt` | .459 | | | | | | |
| `2026-11-20_10-09-13_source_k1_vision_ft/model_15000.pt`（视觉微调版） | synthetic .382 / YOLO 渲染 .380 | | | | | | |

20k 和 vision_ft 两行只做过总挡出率评估，没有按区域统计，所以区域列是空的。

注意：
- **单次评估有噪声**：总挡出率约 ±.03；单个区域相差不到约 .06，可以当成噪声。
- **区域之间会来回摆**：同一次训练里，相邻 checkpoint 的单个区域能差 .15–.3。挑 checkpoint 要评估几个点，不要直接用最后一个。
- **实际挡住的球几乎都碰到了手臂**：没碰到手臂还挡住的情况接近 0，只有低球偶尔例外。

## 3. 常用命令

所有命令都在仓库根目录运行。显卡是 8 GB，训练用 512 个 env，训练时不要同时跑评估。

```bash
# 从 model_12000 继续训练（举例：开启 highreach）
uv run --no-sync train Mjlab-Goalkeeper-Source-Flat-Booster-K1 \
  --gpu-ids '[0]' --env.scene.num-envs 512 \
  --agent.resume True --agent.load-run 2026-11-17_11-28-20_source_k1_upstream_like_cont \
  --agent.load-checkpoint model_12000.pt --agent.max-iterations 3000 \
  --env.rewards.highreach.weight 20.0 --agent.run-name <名字>

# 可视化
uv run --no-sync play Mjlab-Goalkeeper-Source-Flat-Booster-K1 \
  --checkpoint-file logs/rsl_rl/k1_goalkeeper_source/<run>/model_XXXXX.pt --viewer viser

# 按区域统计碰球率、挡出率、手的高度、起跳（推荐用这个评估）
uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.diagnose_reach \
  --checkpoint logs/rsl_rl/k1_goalkeeper_source/<run>/model_XXXXX.pt --num-envs 1200

# 同一个 checkpoint 在不同球输入下的挡出率（truth / synthetic / rendered）
uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.vision.evaluate \
  --checkpoint <ckpt> --ball-input rendered --yolo logs/yolo_ball/k1_head/weights/best.pt --num-envs 600

# 测试（venv 里没有 pytest，直接当脚本跑）
uv run --no-sync python tests/test_goalkeeper_source_flow.py
uv run --no-sync python tests/test_goalkeeper_source_highreach.py
```

视觉微调用 `Mjlab-Goalkeeper-Source-Vision-Flat-Booster-K1`（球输入为 synthetic），
可以直接加载真实输入训练出的 checkpoint 继续训练。
YOLO 数据集和训练见 `source/vision/make_yolo_dataset.py`、`train_yolo.py` 的文件头说明。
YOLO 权重 `logs/yolo_ball/k1_head/weights/best.pt` **没有提交**（ultralytics 是 AGPL 许可）。
需要的话请找我要，或者自己重新训练。

## 4. 近期改动，队友需要知道的

1. **修正了脚底接触力的符号**（`source/state.py` 的 `refresh`）。
   - mjlab 的 ContactSensor 给的是脚压地面的力（z 为负），上游用的是地面反作用力（z 为正）。
   - 修正前，所有 `force_z > 1` 的判断都不成立：起跳检测和 **successland 从来没有发过奖励**。model_12000 就是在这个 bug 下训练出来的。
   - 修正后，successland 开始真正发奖励，见第 5 节的问题 2。
2. **新增 highreach 奖励**（`source/k1_rewards.py` 的 `_reward_highreach`）。
   - 只在 K1 上有，上游没有，默认权重 0。
   - 作用：高球区域里，引导手接近球穿过机器人所在平面时的那个点。
   - 开启方式：`--env.rewards.highreach.weight`。
3. **新增 `source/diagnose_reach.py`**，按区域统计碰球、挡出、手的高度、离球最近距离、起跳，以及 AMP 动作片段本身能达到的手高。
4. 详细过程和数据见 `experiments/goalkeeper/2026-11-21-k1-high-shot-reach.md`。

## 5. 当前问题：高球挡不住

**已经确认的**：
- **不是够不到。** K1 站着把手伸直举过头顶，手能到 1.13 m；AMP 动作片段里，手最高能到 1.23–1.25 m。高球穿过机器人所在平面时的高度中位数是 .99 m。
- **问题出在动作上。** 大约一半没挡住的高球，手从来没举过 .90 m。
- **highreach 加了没用。** 权重 20 和 100 各训了 3000 iter，高球挡出率都 ≤ .05。highreach 的原始分两次完全一样，说明手的动作根本没变。

**目前最可能的原因：successland 在奖励"原地小跳"。**
- successland 是上游原版逻辑，权重 4：起跳之后每一步给 1 分，双脚着地后每一步给 6 分，和球没关系。
- K1 的起跳判定是"双脚离地 2 步 + 躯干升高 5 cm"，基本就是一个小蹦；上游 G1 要求躯干高度超过 1.0 m，大约要升高 20 cm。
- 高球区域每步加权奖励：eereach .31、**successland .145**、highreach（权重 100）.055、stopball .008。
- 现象：修正符号后的三次训练，high-R 的起跳率都升到 .5–.75，碰球率却从 .24 掉到 .02–.06。

**另一个已知问题：stopball 的判定太宽。** 球被碰到、减了速就算挡住，就算最后还是进门了，也能拿到和成功扑救一样的奖励。

## 6. 下一步（按优先级）

1. **对照实验，不改代码**：关掉 successland，保留 highreach 100，从 model_12000 训练 3000 iter。
   ```bash
   ... --env.rewards.highreach.weight 100.0 --env.rewards.successland.weight 0.0 \
       --agent.run-name source_k1_hr100_noland
   ```
   - 结果：高球碰球率回到 .24 以上，并且 TensorBoard 里 `Region/2,3/reward_highreach` 开始上升。
   - 说明：问题就是 successland。
2. **修 successland**（二选一）：
   - 把起跳门槛提到大约 +14 cm（按身高从 G1 等比例换算）。
   - 或者只在手接近球时才给 successland。
3. **收紧 stopball**：改成球没进门，或者 vx 反向，才算挡住。
4. **视觉上真机**：
   - 采集真机相机图片并标注，和仿真图片混在一起训练 YOLO。
   - 标定相机的外参和深度误差，填进 `vision/config.py`。
   - 真机推理链路已经写好：`vision/realtime.py`。

## 7. 坑

- **TensorBoard 的数值是加权后的。** `Region/<区域>/reward_*` = 原始分 × 权重 × 0.6 × 0.02。比较不同权重的训练时，要先换算回原始分。
- **学习率是自适应的。** 配置里写的 1e-3 只是初始值，实际训练中大约在 1e-4。
- **AMP 混合比例固定**：总奖励 = 0.6 × 任务奖励 + 0.4 × 风格奖励，不随训练衰减，和上游一样。
- **区域编号**：0/1 是中路左/右，2/3 是高球左/右，4/5 是低球左/右。
- **lint**：对整个 `source/` 目录跑 ruff 会报 vendor 代码里本来就有的问题，只检查自己改过的文件。
