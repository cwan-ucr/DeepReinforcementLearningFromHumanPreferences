from __future__ import annotations

import argparse
from pathlib import Path

from sumo_rlhf.fcd import load_fcd_trajectories
from sumo_rlhf.sumo_ego_env import SumoEgoConfig, SumoEgoEnv
from sumo_rlhf.trajectory_buffer import StepRecord, TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect passive SUMO expert trajectories for preference learning."
    )
    parser.add_argument("--sumo-cfg", required=True)
    parser.add_argument("--ego-id", default="ego")
    parser.add_argument(
        "--expert-type",
        choices=["sumo-default", "glosa"],
        required=True,
        help="Label stored in each step's info['source'].",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--step-length", type=float, default=0.5)
    parser.add_argument("--segment-length", type=int, default=10)
    parser.add_argument("--output", default="runs/expert_segments.jsonl")
    parser.add_argument("--fcd-output-dir", default="runs/fcd")
    parser.add_argument("--ego-depart-min", type=float, default=0.0)
    parser.add_argument("--ego-depart-max", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use the same SUMO seed for every episode instead of seed+episode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file instead of appending to it.",
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
            ego_type_id="ego_glosa_type" if args.expert_type == "glosa" else "ego_type",
            ego_depart_min=args.ego_depart_min,
            ego_depart_max=args.ego_depart_max,
        )
    )

    buffer = TrajectoryBuffer()
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            episode_steps = []

            for _ in range(args.max_steps):
                next_obs, _reward, done, info = env.passive_step()
                action_value = float(info["action_accel"])
                info["source"] = args.expert_type

                episode_steps.append(
                    StepRecord(
                        obs=obs.tolist(),
                        action_index=-1,
                        action_value=float(action_value),
                        next_obs=next_obs.tolist(),
                        done=done,
                        info=info,
                    )
                )
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
                segment_prefix=args.expert_type,
                traffic_trajectories=traffic_trajectories,
            )
            depart_time = (
                float(episode_steps[0].info.get("ego_depart_time"))
                if episode_steps
                else float("nan")
            )
            print(
                f"expert={args.expert_type} episode={episode} "
                f"steps={len(episode_steps)} seed={episode_steps[0].info.get('sumo_seed') if episode_steps else None} "
                f"depart={depart_time:.1f}"
            )
    finally:
        env.close()

    output = Path(args.output)
    existing = (
        TrajectoryBuffer.load_jsonl(output)
        if output.exists() and not args.overwrite
        else TrajectoryBuffer()
    )
    existing.segments.extend(buffer.segments)
    existing.save_jsonl(output)
    mode = "saved" if args.overwrite else "saved/appended"
    print(f"{mode} {len(buffer.segments)} expert segments to {output}")


if __name__ == "__main__":
    main()
