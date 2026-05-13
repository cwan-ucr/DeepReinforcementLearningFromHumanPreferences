# SUMO RLHF 单车道干道轨迹优化

这是在现有 CartPole RLHF demo 旁边新增的一套 SUMO 项目骨架，目标是：

> 在单车道城市干道 SUMO 场景中，让 RL agent 控制一辆 ego vehicle 的纵向加速度，通过人类对轨迹片段的偏好比较学习 reward model，再用 learned reward 训练策略。

## 第一版范围

- 场景：城市干道，可包含多个连续信号灯。
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

生成这个三交叉口单车道干道场景：

```bash
python scripts/create_simple_arterial.py --build
```

这会在 `scenarios/simple_arterial/` 下生成：

```text
simple_arterial.net.xml
simple_arterial.rou.xml
simple_arterial.tls.xml
simple_arterial.sumocfg
```

场景参数：

```text
交叉口数量: 3
交叉口间距: 300 m
路段总长: 1200 m
车道数: 单车道
信号周期: 90 s
绿灯: 45 s
红灯: 45 s
三个信号灯 offset: 0 s
ego vehicle id: ego
```

收集轨迹片段：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 20 \
  --segment-length 10 \
  --output runs/sumo_segments.jsonl
```

人工标注偏好：

```bash
python label_preferences.py \
  --segments runs/sumo_segments.jsonl \
  --output runs/preferences.jsonl \
  --pairs 50
```

训练 reward model：

```bash
python train_reward_model.py \
  --segments runs/sumo_segments.jsonl \
  --preferences runs/preferences.jsonl \
  --output runs/reward_model.pt
```

加载 learned reward 继续训练策略：

```bash
python train_sumo_rlhf.py \
  --sumo-cfg scenarios/simple_arterial/simple_arterial.sumocfg \
  --ego-id ego \
  --episodes 100 \
  --reward-checkpoint runs/reward_model.pt
```

## 下一步

第一版 `label_preferences.py` 只显示轨迹统计摘要，适合验证数据链路。真正用于人类驾驶偏好时，下一步应该把每个 segment 渲染为并排视频或速度-距离-信号灯时序图，让人类直接比较驾驶过程。
