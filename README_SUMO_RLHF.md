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
[-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]  # m/s^2
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
  -> synthetic expert-ranking preferences
  -> pretrained reward ensemble
  -> first RL policy rollout
  -> human pairwise preferences focused on RL-vs-expert comparisons
  -> reward ensemble r_theta(obs, action)
  -> PPO policy training with learned reward
```

当前代码包含：

- `sumo_rlhf/sumo_ego_env.py`：SUMO ego vehicle Gym-style 环境。
- `sumo_rlhf/ppo_agent.py`：离散动作 PPO，用当前 rollout 进行 on-policy 更新。
- `sumo_rlhf/trajectory_buffer.py`：保存固定长度轨迹片段。
- `sumo_rlhf/reward_model.py`：轨迹级 Bradley-Terry 偏好奖励模型。
- `generate_preference_labels.py`：按专家排序自动生成预训练偏好标签。
- `label_preferences.py`：终端版偏好标注工具。
- `train_reward_model.py`：从偏好标签训练 reward model，支持 ensemble、dropout 和 L2 正则。
- `train_sumo_rlhf.py`：用 SUMO rollout 收集片段，并可加载 learned reward 训练策略。

## 运行方式

也可以使用统一 pipeline 脚本，减少手动输入命令。

如果想设定迭代轮数并自动跑完整闭环，使用：

```bash
python run_pipeline.py \
  --stage iterate \
  --rounds 3 \
  --pairs 50 \
  --match-mode scene \
  --whole-episode-segments \
  --ego-depart-min 0 \
  --ego-depart-max 90
```

这个命令会执行：

```text
setup
collect expert/random trajectories
generate synthetic preferences: glosa > sumo-default > rl-random
pretrain reward ensemble
train bootstrap RL policy from pretrained reward
merge bootstrap RL policy into the first human-labeling pool
round 1: human label -> reward ensemble -> RLHF policy -> next pool
round 2: human label -> reward ensemble -> RLHF policy -> next pool
round 3: human label -> reward ensemble -> RLHF policy
```

第一轮不需要人工标注，会先用专家排序伪标签预训练奖励模型，再用这个预训练 reward 更新出一版 bootstrap PPO 策略，并把这批 `rl-policy` 轨迹合并进第一轮人工标注池。这样人工标注看到的是已经被预训练 reward 拉开差异的 RL 轨迹，而不是只在专家/random 之间重复打标签。

之后每轮网页标注达到 `--pairs` 后会自动关闭，pipeline 会继续进入下一阶段。默认人工标注 pair 的主要部分是 `rl-policy` 自己和自己比较，用来在当前策略分布内部学习“哪个 RL 轨迹更像人类偏好”；同时保留少量 `rl-policy` vs 专家/随机、专家 vs 专家/随机，防止 reward model 跟着当前策略跑偏。加 `--whole-episode-segments` 后，每个 episode 会作为一条完整轨迹参与比较，网页动画也会播放完整轨迹而不是固定 10 秒窗口。每轮产物会自动命名：

默认 `--match-mode scene` 会先把轨迹片段按场景分层抽样：起始位置在 0-350m 内每 50m 分一个空间区域，信号周期 0-90s 等距分成 4 个相位区域，因此一共有 7 × 4 = 28 个场景。采样时优先在同一个 `位置区域 × 信号相位区域` 内抽 pair，再套用 source-pair 权重。如果使用整条轨迹比较，更推荐加 `--match-mode episode`，让左右两侧优先来自同一个 episode；pipeline 在这个模式下会让专家、random、policy 使用同一个 SUMO seed 序列，使相同 episode 具有相同背景交通和 ego depart 初始化。policy 阶段默认每个场景 episode 采样 2 条 `rl-policy` 轨迹，因此 buffer 里的 policy 轨迹数会变成 `policy_episodes × policy_rollouts_per_episode`。

```text
runs/pretrain_preferences.jsonl
runs/reward_model_pretrain.pt
runs/rlhf_policy_segments_round0.jsonl
runs/ppo_policy_round0.pt
runs/preferences_round1.jsonl
runs/reward_model_round1.pt
runs/rlhf_policy_segments_round1.jsonl
runs/ppo_policy_round1.pt
runs/preference_pool_round2.jsonl
...
```

默认 reward model 是 5 个成员的 ensemble，并使用 `dropout=0.1` 和 `weight_decay=1e-4`。可以通过下面参数调整：

```bash
python run_pipeline.py \
  --stage iterate \
  --ensemble-size 5 \
  --reward-dropout 0.1 \
  --reward-weight-decay 1e-4
```

策略训练现在使用 PPO，而不是 DQN replay buffer。PPO 只用当前 rollout 更新策略；每次 `train_sumo_rlhf.py` 启动时 reward model 固定，rollout 内 reward 由当前 reward model 现场计算。常用 PPO 参数也可以从 pipeline 传入：
pipeline 会保存 `runs/ppo_policy_round*.pt`，下一轮会从上一轮策略 warm start，然后用新的 reward model 继续 on-policy 更新。

```bash
python run_pipeline.py \
  --stage iterate \
  --ppo-learning-rate 3e-4 \
  --ppo-clip-ratio 0.2 \
  --ppo-update-epochs 4 \
  --ppo-entropy-coef 0.01
```

如果希望在同一个外部场景下比较多条 RL policy 采样轨迹，可以调大：

```bash
python run_pipeline.py \
  --stage iterate \
  --whole-episode-segments \
  --match-mode episode \
  --policy-episodes 80 \
  --policy-rollouts-per-episode 2
```

这里 `policy-episodes` 表示不同的外部场景数量，`policy-rollouts-per-episode` 表示每个场景下重复采样几条 RL policy 轨迹。每条重复轨迹的 `episode_id` 相同，但 segment id 会带 `r00/r01`，例如 `rl-policy_r00_ep00005_full` 和 `rl-policy_r01_ep00005_full`。PPO 会先采完同一个 episode 的所有重复 rollout，再做一次 on-policy 更新，避免第二条轨迹使用已经被第一条轨迹更新过的 policy。

默认人工标注 source-pair 配比是：

```text
rl-policy vs rl-policy    65%
rl-policy vs glosa        12%
rl-policy vs sumo-default 10%
rl-policy vs rl-random     8%
glosa vs sumo-default      3%
sumo-default vs rl-random  2%
```

这个配比可以改：

```bash
python run_pipeline.py \
  --stage label \
  --human-source-pair-weights "rl-policy:rl-policy:0.65,rl-policy:glosa:0.12,rl-policy:sumo-default:0.10,rl-policy:rl-random:0.08,glosa:sumo-default:0.03,sumo-default:rl-random:0.02"
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

如果想直接比较整条轨迹，可以加：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 20 \
  --step-length 0.5 \
  --whole-episode-segments \
  --output runs/sumo_segments.jsonl
```

此时每个 episode 只生成一个 segment，segment id 类似 `rl-random_ep00000_full` 或 `rl-policy_ep00000_full`。

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

每次抽取一对轨迹片段时，脚本默认使用 `scene` 匹配，并优先在同一个 28 场景分层内抽取轨迹片段。这样可以让两条轨迹处在相近的道路位置和信号相位背景下，再比较 `rl-policy`、`rl-random`、`sumo-default`、`glosa` 之间的差异。

```text
默认 match-mode: scene
空间：0-350m 每 50m 分一份
信号相位：90s 周期等距分 4 份
共 28 个场景
```

可以按需要切换匹配方式：

```bash
python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --match-mode episode

python preference_web.py \
  --segments runs/preference_pool.jsonl \
  --output runs/preferences.jsonl \
  --match-mode scene

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

脚本默认会在网页里用 canvas 生成左右并排的道路视角实时动图，时间窗口固定为 10 秒，默认按 1.5 倍速度播放。若设置 `--animation-window-seconds 0`，则播放完整 segment；pipeline 在 `--whole-episode-segments` 时会自动使用完整轨迹窗口。主道路窗口是 ego 前后各 50m 的动态局部视角，ego 车身中心固定在画面中心；蓝色车辆是 ego，灰色车辆是背景车。SUMO/FCD 的 `position` 按车头位置处理，绘图时车辆按车身长度从车头向后延伸，而不是把车头位置误当成车身中心。动画上方会显示两侧片段的辅助评价指标，包括平均速度、百公里能耗、累计 TET、累计 jerk 和停车时间。道路上方左侧显示速度/加速度随时间变化的小曲线，其中 RL 轨迹会同时显示紫色 `cmd` 加速度和橙色虚线 `actual` 加速度；右侧显示完整 90 秒信号周期和当前相位滑动竖线，并用一个小框显示 ego 车头在当前路口/信号灯附近的全局位置。下方同时显示当前速度、加速度、前车距离和后车距离。

也可以单独导出轨迹评价指标：

```bash
python evaluate_segments.py \
  --segments runs/preference_pool.jsonl \
  --output runs/evaluation.csv
```

如果已经训练了 reward model，可以同时输出 learned reward score：

```bash
python evaluate_segments.py \
  --segments runs/preference_pool.jsonl \
  --reward-checkpoint runs/reward_model.pt \
  --output runs/evaluation.csv
```

脚本会同时生成逐片段指标和按 `source` 聚合的 summary CSV。

能耗指标优先使用 TraCI 每一步返回的 SUMO 能耗值：`vehicle.getFuelConsumption()` 和 `vehicle.getElectricityConsumption()`。评价时先积分得到总能耗，再按 segment 行驶距离换算为 `energy_kwh_per_100km`。旧轨迹如果没有保存 TraCI 能耗字段，评价脚本会退回到速度/加速度近似能耗模型。

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
  --preferences runs/pretrain_preferences.jsonl runs/preferences_round1.jsonl \
  --output runs/reward_model.pt \
  --ensemble-size 5 \
  --dropout 0.1 \
  --weight-decay 1e-4
```

加载 learned reward 继续训练策略：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 100 \
  --ego-depart-min 0 \
  --ego-depart-max 90 \
  --reward-checkpoint runs/reward_model.pt \
  --ppo-learning-rate 3e-4 \
  --ppo-clip-ratio 0.2
```

## 下一步

当前已经可以用网页对轨迹图做偏好标注。后续如果想进一步提升标注质量，可以把每个 segment 渲染成并排视频，让标注者直接看完整驾驶过程。
