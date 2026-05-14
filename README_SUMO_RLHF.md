# SUMO RLHF 单车道干道轨迹优化

这是在现有 CartPole RLHF demo 旁边新增的一套 SUMO 项目骨架，目标是：

> 在单车道城市干道 SUMO 场景中，让 RL agent 控制一辆 ego vehicle 的纵向加速度，通过人类对轨迹片段的偏好比较学习 reward model，再用 learned reward 训练策略。

## 第一版范围

- 场景：第一版先使用一个信号路口，后续可扩展到多个连续信号灯。
- 道路：单车道。
- 控制对象：一辆 ego vehicle。
- 控制动作：纵向加速度，不考虑换道。
- 不处理碰撞和闯红灯约束，假设 SUMO 场景或车辆模型已经保证。

## 观测

`SumoEgoEnv` 的观测向量在 `sumo_rlhf/sumo_ego_env.py` 中定义：

```text
[
  position,
  speed,
  front_distance,
  ego_minus_front_speed,
  previous_action_accel,
  next_tls_distance,
  next_tls_time_remaining,
  tls_red,
  tls_yellow,
  tls_green
]
```

默认会归一化到大致 `[-1, 1]` 范围。原始物理量会保存在每一步的 `info["raw_observation"]` 里，方便做人类偏好摘要或后续可视化。

## 动作

默认离散动作：

```text
[-3.0, -1.5, 0.0, 1.0, 2.0]  # m/s^2
```

训练脚本会把动作转换为下一步速度指令：

```text
v_next = clip(v_current + accel * dt, 0, speed_limit)
```

并通过 TraCI 的 `vehicle.setSpeed()` 控制 ego 车。

## 数据流

```text
SUMO rollout
  -> trajectory segments
  -> human pairwise preferences
  -> reward model r_theta(obs, action)
  -> DQN policy training with learned reward
```

当前代码包含：

- `sumo_rlhf/sumo_ego_env.py`：SUMO ego vehicle Gym-style 环境。
- `sumo_rlhf/dqn_agent.py`：第一版离散动作 DQN。
- `sumo_rlhf/trajectory_buffer.py`：保存固定长度轨迹片段。
- `sumo_rlhf/reward_model.py`：轨迹级 Bradley-Terry 偏好奖励模型。
- `label_preferences.py`：终端版偏好标注工具。
- `train_reward_model.py`：从偏好标签训练 reward model。
- `train_sumo_rlhf.py`：用 SUMO rollout 收集片段，并可加载 learned reward 训练策略。

## 运行方式

也可以使用统一 pipeline 脚本，减少手动输入命令。

如果想设定迭代轮数并自动跑完整闭环，使用：

```bash
python run_pipeline.py \
  --stage iterate \
  --rounds 3 \
  --pairs 50 \
  --match-mode time \
  --ego-depart-min 0 \
  --ego-depart-max 90
```

这个命令会执行：

```text
setup
collect expert/random trajectories
round 1: label -> reward model -> RLHF policy -> next pool
round 2: label -> reward model -> RLHF policy -> next pool
round 3: label -> reward model -> RLHF policy
```

每轮网页标注达到 `--pairs` 后会自动关闭，pipeline 会继续进入下一阶段。每轮产物会自动命名：

```text
runs/preferences_round1.jsonl
runs/reward_model_round1.pt
runs/rlhf_policy_segments_round1.jsonl
runs/preference_pool_round2.jsonl
...
```

生成场景、采集 expert/random 轨迹、合并偏好池：

```bash
python run_pipeline.py --stage all-before-label
```

网页标注：

```bash
python run_pipeline.py \
  --stage label \
  --preferences runs/preferences_round1.jsonl \
  --pairs 50 \
  --match-mode time
```

标注完成后，训练 reward model、训练 RLHF policy，并生成下一轮偏好池：

```bash
python run_pipeline.py \
  --stage all-after-label \
  --preferences runs/preferences_round1.jsonl
```

也可以只执行某个阶段：

```bash
python run_pipeline.py --stage setup
python run_pipeline.py --stage collect
python run_pipeline.py --stage reward --preferences runs/preferences_round1.jsonl
python run_pipeline.py --stage policy
python run_pipeline.py --stage round2
```

生成这个单路口单车道场景：

```bash
python scripts/create_simple_arterial.py --build
```

这会在 `scenarios/simple_arterial/` 下生成：

```text
simple_arterial.net.xml
simple_arterial.rou.xml
simple_arterial_glosa.rou.xml
simple_arterial.tls.xml
simple_arterial.sumocfg
simple_arterial_glosa.sumocfg
```

场景参数：

```text
交叉口数量: 1
上游长度: 300 m
下游长度: 50 m
路段总长: 350 m
车道数: 单车道
仿真步长: 0.5 s
信号周期: 90 s
绿灯: 45 s
红灯: 45 s
ego vehicle id: ego
ego 出发时间: 每个 episode 在 0-90 s 内随机采样
背景车流: 随机到达，平均约每 8 秒一辆
```

路由文件里不再固定写入 `<vehicle id="ego">`。每次 `env.reset()` 时，代码会先让 SUMO 背景交通运行到随机采样的 ego 出发时间，再通过 TraCI 把 ego 插入到 `arterial_route`。因此同一个策略会看到不同信号相位和不同背景车前后关系。

收集轨迹片段：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 20 \
  --step-length 0.5 \
  --segment-length 20 \
  --seed 42 \
  --ego-depart-min 0 \
  --ego-depart-max 90 \
  --output runs/sumo_segments.jsonl
```

这里 `--segment-length` 是 step 数，不是秒数。`step-length=0.5` 且 `segment-length=20` 表示每个偏好片段约 10 秒。

`--ego-depart-min/--ego-depart-max` 控制 ego 出发时间的随机范围。当前默认是 `0-90s`，对应一个完整信号周期；如果你想让 ego 更集中地在红灯或绿灯附近出现，可以把这个范围缩小。

默认每个 episode 会使用不同 SUMO seed：`seed, seed+1, seed+2, ...`，因此背景交通流和 ego 出发时间都会变化。若要完全固定每个 episode 的交通流和 ego 出发时间，可以加 `--fixed-seed`。

收集 SUMO 默认跟车模型专家轨迹：

```bash
python collect_expert_trajectories.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --expert-type sumo-default \
  --episodes 20 \
  --step-length 0.5 \
  --segment-length 20 \
  --seed 42 \
  --ego-depart-min 0 \
  --ego-depart-max 90 \
  --overwrite \
  --output runs/expert_segments.jsonl
```

收集 GLOSA 专家轨迹：

```bash
python collect_expert_trajectories.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial_glosa.sumocfg \
  --ego-id ego \
  --expert-type glosa \
  --episodes 20 \
  --step-length 0.5 \
  --segment-length 20 \
  --seed 42 \
  --ego-depart-min 0 \
  --ego-depart-max 90 \
  --output runs/expert_segments.jsonl
```

这两类专家轨迹都是 passive rollout：ego 由 SUMO 自带模型驾驶，代码只记录状态和实际加速度。默认专家来自 SUMO car-following model，GLOSA 专家来自 SUMO 的 `glosa` device。

合并 RL 轨迹和专家轨迹，形成偏好标注池：

```bash
python merge_trajectory_segments.py \
  --inputs runs/sumo_segments.jsonl runs/expert_segments.jsonl \
  --output runs/preference_pool.jsonl
```

人工标注偏好：

```bash
python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --plot-dir runs/preference_web_plots \
  --pairs 50 \
  --match-mode time \
  --media road \
  --playback-speed 1.5
```

打开脚本打印出的本地网页地址后，可以直接在页面里看左右对比图并点击按钮标注。快捷键：

```text
1: 左边更好
2: 右边更好
3 或 0: 差不多
```

如果不想开网页，也可以继续使用终端版：

```bash
python label_preferences.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --plot-dir runs/preference_plots \
  --pairs 50
```

每次抽取一对轨迹片段时，脚本默认只按仿真时间匹配，并优先选择不同来源的轨迹片段。这样可以在类似信号相位背景下比较 `rl-random`、`sumo-default`、`glosa` 之间的差异，而不是总拿同一类轨迹互相比。

```text
默认 match-mode: time
默认起始仿真时间差 <= 20 s
默认优先不同来源 pair
```

可以按需要切换匹配方式：

```bash
python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --match-mode position \
  --match-position-tol 20

python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --match-mode random

python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --match-mode both \
  --match-position-tol 20 \
  --match-time-tol 10
```

如果你确实想允许同来源互相比较，可以加 `--allow-same-source`。

脚本默认会在网页里用 canvas 生成左右并排的道路视角实时动图，时间窗口固定为 10 秒，默认按 1.5 倍速度播放。ego 是蓝色车辆，背景车是灰色车辆；车辆按车身长度映射到道路横轴比例，并居中画在道路中心线上。道路上方左侧显示速度/加速度随时间变化的小曲线，其中 RL 轨迹会同时显示紫色 `cmd` 加速度和橙色虚线 `actual` 加速度；右侧显示完整 90 秒信号周期和当前相位滑动竖线。下方同时显示当前速度、加速度、前车距离和后车距离。

如果还想看原来的曲线动态图，可以加：

```bash
--media interactive
```

曲线动态图包含：

```text
位置-时间图: ego 位置、该时间窗口内所有背景车辆轨迹、信号灯状态
速度-时间图
加速度指令-时间图
前后车距离-时间图
```

如果想回到静态 PNG，或者导出 GIF 文件，可以加：

```bash
--media static
--media animation
```

也可以单独画某个片段或某一对片段：

```bash
python plot_trajectory_segments.py \
  --segments runs/sumo_segments.jsonl \
  --segment-id ep00000_t00000

python plot_trajectory_segments.py \
  --segments runs/sumo_segments.jsonl \
  --left-id ep00000_t00000 \
  --right-id ep00001_t00000
```

单独生成 GIF 动图：

```bash
python plot_trajectory_segments.py \
  --segments runs/preference_pool.jsonl \
  --left-id rl-random_ep00000_t00000 \
  --right-id sumo-default_ep00000_t00000 \
  --animate \
  --animation-window-seconds 10 \
  --animation-fps 4
```

训练 reward model：

```bash
python train_reward_model.py \
  --segments runs/preference_pool.jsonl \
  --preferences runs/preferences.jsonl \
  --output runs/reward_model.pt
```

加载 learned reward 继续训练策略：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 100 \
  --ego-depart-min 0 \
  --ego-depart-max 90 \
  --reward-checkpoint runs/reward_model.pt
```

## 下一步

当前已经可以用网页对轨迹图做偏好标注。后续如果想进一步提升标注质量，可以把每个 segment 渲染成并排视频，让标注者直接看完整驾驶过程。
