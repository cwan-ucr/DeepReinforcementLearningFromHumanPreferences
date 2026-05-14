from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sumo_rlhf.dqn_agent import DQNAgent
from sumo_rlhf.fcd import load_fcd_trajectories
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
    parser.add_argument("--step-length", type=float, default=0.5)
    parser.add_argument("--segment-length", type=int, default=10)
    parser.add_argument("--output", default="runs/sumo_segments.jsonl")
    parser.add_argument("--fcd-output-dir", default="runs/fcd")
    parser.add_argument("--ego-depart-min", type=float, default=0.0)
    parser.add_argument("--ego-depart-max", type=float, default=90.0)
    parser.add_argument("--reward-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use the same SUMO seed for every episode instead of seed+episode.",
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = SumoEgoEnv(
        SumoEgoConfig(
            sumo_cfg=args.sumo_cfg,
            ego_id=args.ego_id,
            gui=args.gui,
            step_length=args.step_length,
            max_episode_steps=args.max_steps,
            seed=args.seed,
            randomize_seed_on_reset=not args.fixed_seed,
            fcd_output_dir=args.fcd_output_dir,
            ego_type_id="ego_type",
            ego_depart_min=args.ego_depart_min,
            ego_depart_max=args.ego_depart_max,
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
        print(f"loaded reward model from {args.reward_checkpoint}; training policy")
    else:
        print("no reward checkpoint provided; collecting random trajectory segments only")

    buffer = TrajectoryBuffer()
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            episode_steps = []
            total_reward = 0.0

            for _ in range(args.max_steps):
                action = (
                    agent.act(obs)
                    if reward_model is not None
                    else int(env.action_space.sample())
                )
                next_obs, _env_reward, done, info = env.step(action)
                action_value = float(info["action_accel"])

                if reward_model is None:
                    reward = 0.0
                else:
                    reward = reward_model.predict_step_reward(obs.tolist(), action_value)

                if reward_model is not None:
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

            fcd_path = env.fcd_output_path
            env.close()
            traffic_trajectories = load_fcd_trajectories(fcd_path, ego_id=args.ego_id)
            buffer.add_episode(
                episode_id=episode,
                episode_steps=episode_steps,
                segment_length=args.segment_length,
                segment_prefix="rl-policy" if reward_model is not None else "rl-random",
                traffic_trajectories=traffic_trajectories,
            )
            depart_time = (
                float(episode_steps[0].info.get("ego_depart_time"))
                if episode_steps
                else float("nan")
            )
            print(
                f"episode={episode} steps={len(episode_steps)} "
                f"learned_reward={total_reward:.3f} "
                f"epsilon={agent.exploration_rate:.3f} "
                f"seed={episode_steps[0].info.get('sumo_seed') if episode_steps else None} "
                f"depart={depart_time:.1f}"
            )
    finally:
        env.close()

    output = Path(args.output)
    buffer.save_jsonl(output)
    print(f"saved {len(buffer.segments)} trajectory segments to {output}")


if __name__ == "__main__":
    main()
