from __future__ import annotations

import argparse
from pathlib import Path

from sumo_rlhf.fcd import load_fcd_trajectories
from sumo_rlhf.ppo_agent import PPOAgent, PPOConfig
from sumo_rlhf.reward_model import load_reward_checkpoint
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
    parser.add_argument(
        "--whole-episode-segments",
        action="store_true",
        help="Store each episode as one full-trajectory segment instead of fixed-length slices.",
    )
    parser.add_argument(
        "--rollouts-per-episode",
        type=int,
        default=1,
        help=(
            "Repeat each scenario episode this many times with the same SUMO seed. "
            "Useful for collecting multiple stochastic RL-policy trajectories under "
            "identical traffic initialization."
        ),
    )
    parser.add_argument("--output", default="runs/sumo_segments.jsonl")
    parser.add_argument("--fcd-output-dir", default="runs/fcd")
    parser.add_argument("--ego-depart-min", type=float, default=0.0)
    parser.add_argument("--ego-depart-max", type=float, default=90.0)
    parser.add_argument("--reward-checkpoint", default=None)
    parser.add_argument("--policy-checkpoint-in", default=None)
    parser.add_argument("--policy-checkpoint-out", default=None)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use the same SUMO seed for every episode instead of seed+episode.",
    )
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def scenario_seed_for_episode(args, episode: int):
    if args.seed is None:
        return None
    if args.fixed_seed:
        return int(args.seed)
    return int(args.seed) + int(episode)


def main():
    args = parse_args()
    if args.rollouts_per_episode < 1:
        raise ValueError("--rollouts-per-episode must be >= 1.")
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
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_count=env.action_space.n,
        config=PPOConfig(
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            learning_rate=args.ppo_learning_rate,
            clip_ratio=args.ppo_clip_ratio,
            update_epochs=args.ppo_update_epochs,
            batch_size=args.ppo_batch_size,
            entropy_coef=args.ppo_entropy_coef,
            value_coef=args.ppo_value_coef,
        ),
    )

    reward_model = None
    if args.reward_checkpoint:
        reward_model = load_reward_checkpoint(
            args.reward_checkpoint,
            obs_dim=env.observation_space.shape[0],
        )
        print(f"loaded reward model from {args.reward_checkpoint}; training policy")
        if args.policy_checkpoint_in:
            agent.load(args.policy_checkpoint_in)
            print(f"loaded PPO policy from {args.policy_checkpoint_in}")
    else:
        print("no reward checkpoint provided; collecting random trajectory segments only")

    buffer = TrajectoryBuffer()
    try:
        for episode in range(args.episodes):
            update_metrics = None
            rollout_records = []
            scenario_seed = scenario_seed_for_episode(args, episode)

            for rollout_index in range(args.rollouts_per_episode):
                fcd_episode_index = episode * args.rollouts_per_episode + rollout_index
                obs = env.reset(
                    scenario_seed=scenario_seed,
                    fcd_episode_index=fcd_episode_index,
                )
                episode_steps = []
                total_reward = 0.0

                for _ in range(args.max_steps):
                    if reward_model is not None:
                        action, log_prob, value = agent.act(obs)
                    else:
                        action = int(env.action_space.sample())
                        log_prob = 0.0
                        value = 0.0
                    next_obs, _env_reward, done, info = env.step(action)
                    action_value = float(info["action_accel"])

                    if reward_model is None:
                        reward = 0.0
                    else:
                        reward = reward_model.predict_step_reward(obs.tolist(), action_value)

                    if reward_model is not None:
                        agent.remember(obs, action, log_prob, value, reward, done)

                    source = "rl-policy" if reward_model is not None else "rl-random"
                    info["source"] = source
                    info["scenario_episode_id"] = episode
                    info["policy_rollout_index"] = rollout_index

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
                source = "rl-policy" if reward_model is not None else "rl-random"
                segment_prefix = (
                    source
                    if args.rollouts_per_episode == 1
                    else f"{source}_r{rollout_index:02d}"
                )
                buffer.add_episode(
                    episode_id=episode,
                    episode_steps=episode_steps,
                    segment_length=args.segment_length,
                    segment_prefix=segment_prefix,
                    traffic_trajectories=traffic_trajectories,
                    whole_episode=args.whole_episode_segments,
                )
                rollout_records.append((rollout_index, episode_steps, total_reward))

            if reward_model is not None:
                update_metrics = agent.update(next_value=0.0)
            update_text = ""
            if update_metrics:
                update_text = (
                    f"ppo_loss={update_metrics['loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.3f} "
                )
            for rollout_index, episode_steps, total_reward in rollout_records:
                depart_time = (
                    float(episode_steps[0].info.get("ego_depart_time"))
                    if episode_steps
                    else float("nan")
                )
                print(
                    f"episode={episode} rollout={rollout_index} "
                    f"steps={len(episode_steps)} "
                    f"learned_reward={total_reward:.3f} "
                    f"{update_text}"
                    f"seed={episode_steps[0].info.get('sumo_seed') if episode_steps else None} "
                    f"depart={depart_time:.1f}"
                )
    finally:
        env.close()

    output = Path(args.output)
    buffer.save_jsonl(output)
    print(f"saved {len(buffer.segments)} trajectory segments to {output}")
    if reward_model is not None and args.policy_checkpoint_out:
        agent.save(args.policy_checkpoint_out)
        print(f"saved PPO policy to {args.policy_checkpoint_out}")


if __name__ == "__main__":
    main()
