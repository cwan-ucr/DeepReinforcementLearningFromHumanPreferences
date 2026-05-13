from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sumo_rlhf.dqn_agent import DQNAgent
from sumo_rlhf.reward_model import StepRewardModel
from sumo_rlhf.sumo_ego_env import SumoEgoConfig, SumoEgoEnv
from sumo_rlhf.trajectory_buffer import StepRecord, TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect/train a single-lane SUMO ego-vehicle policy with RLHF rewards."
    )
    parser.add_argument("--sumo-cfg", required=True, help="Path to the SUMO .sumocfg file.")
    parser.add_argument("--ego-id", default="ego", help="SUMO vehicle id controlled by the agent.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--segment-length", type=int, default=10)
    parser.add_argument("--output", default="runs/sumo_segments.jsonl")
    parser.add_argument("--reward-checkpoint", default=None)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = SumoEgoEnv(
        SumoEgoConfig(
            sumo_cfg=args.sumo_cfg,
            ego_id=args.ego_id,
            gui=args.gui,
            max_episode_steps=args.max_steps,
        )
    )
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        action_count=env.action_space.n,
    )

    reward_model = None
    if args.reward_checkpoint:
        reward_model = StepRewardModel(obs_dim=env.observation_space.shape[0])
        reward_model.load_state_dict(torch.load(args.reward_checkpoint, map_location="cpu"))
        reward_model.eval()

    buffer = TrajectoryBuffer()
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            episode_steps = []
            total_reward = 0.0

            for _ in range(args.max_steps):
                action = agent.act(obs)
                next_obs, _env_reward, done, info = env.step(action)
                action_value = float(info["action_accel"])

                if reward_model is None:
                    reward = 0.0
                else:
                    reward = reward_model.predict_step_reward(obs.tolist(), action_value)

                agent.remember(obs, action, reward, next_obs, done)
                agent.update()

                episode_steps.append(
                    StepRecord(
                        obs=obs.tolist(),
                        action_index=action,
                        action_value=action_value,
                        next_obs=next_obs.tolist(),
                        done=done,
                        info=info,
                    )
                )
                total_reward += reward
                obs = next_obs

                if done:
                    break

            buffer.add_episode(
                episode_id=episode,
                episode_steps=episode_steps,
                segment_length=args.segment_length,
            )
            print(
                f"episode={episode} steps={len(episode_steps)} "
                f"learned_reward={total_reward:.3f} "
                f"epsilon={agent.exploration_rate:.3f}"
            )
    finally:
        env.close()

    output = Path(args.output)
    buffer.save_jsonl(output)
    print(f"saved {len(buffer.segments)} trajectory segments to {output}")


if __name__ == "__main__":
    main()
