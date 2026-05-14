from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]):
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SUMO RLHF workflow.")
    parser.add_argument(
        "--stage",
        choices=[
            "setup",
            "collect",
            "label",
            "reward",
            "policy",
            "round2",
            "iterate",
            "all-before-label",
            "all-after-label",
        ],
        required=True,
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy-episodes", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--step-length", type=float, default=0.5)
    parser.add_argument("--segment-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl-seed", type=int, default=100)
    parser.add_argument("--policy-seed", type=int, default=200)
    parser.add_argument("--ego-depart-min", type=float, default=0.0)
    parser.add_argument("--ego-depart-max", type=float, default=90.0)
    parser.add_argument("--sumo-cfg", default="scenarios/simple_arterial/simple_arterial.sumocfg")
    parser.add_argument(
        "--glosa-cfg",
        default="scenarios/simple_arterial/simple_arterial_glosa.sumocfg",
    )
    parser.add_argument("--expert-segments", default="runs/expert_segments.jsonl")
    parser.add_argument("--random-segments", default="runs/sumo_segments.jsonl")
    parser.add_argument("--preference-pool", default="runs/preference_pool.jsonl")
    parser.add_argument("--preferences", default="runs/preferences_round1.jsonl")
    parser.add_argument("--reward-model", default="runs/reward_model.pt")
    parser.add_argument("--policy-segments", default="runs/rlhf_policy_segments.jsonl")
    parser.add_argument("--round2-pool", default="runs/preference_pool_round2.jsonl")
    parser.add_argument("--plot-dir", default="runs/preference_web_plots")
    parser.add_argument("--match-mode", choices=["time", "position", "both", "random"], default="time")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def exists(path: str) -> bool:
    return Path(path).exists()


def maybe_run(command: list[str], outputs: list[str], skip_existing: bool):
    if skip_existing and outputs and all(exists(output) for output in outputs):
        print(f"skip existing: {', '.join(outputs)}")
        return
    run(command)


def setup(args):
    run([sys.executable, "scripts/create_simple_arterial.py", "--build"])


def collect(args):
    if args.skip_existing and exists(args.expert_segments):
        print(f"skip existing: {args.expert_segments}")
    else:
        run(
            [
                sys.executable,
                "collect_expert_trajectories.py",
                "--sumo-cfg",
                args.sumo_cfg,
                "--ego-id",
                "ego",
                "--expert-type",
                "sumo-default",
                "--episodes",
                str(args.episodes),
                "--step-length",
                str(args.step_length),
                "--segment-length",
                str(args.segment_length),
                "--seed",
                str(args.seed),
                "--ego-depart-min",
                str(args.ego_depart_min),
                "--ego-depart-max",
                str(args.ego_depart_max),
                "--overwrite",
                "--output",
                args.expert_segments,
                "--fcd-output-dir",
                "runs/fcd_expert_default",
            ]
        )
        run(
            [
                sys.executable,
                "collect_expert_trajectories.py",
                "--sumo-cfg",
                args.glosa_cfg,
                "--ego-id",
                "ego",
                "--expert-type",
                "glosa",
                "--episodes",
                str(args.episodes),
                "--step-length",
                str(args.step_length),
                "--segment-length",
                str(args.segment_length),
                "--seed",
                str(args.seed),
                "--ego-depart-min",
                str(args.ego_depart_min),
                "--ego-depart-max",
                str(args.ego_depart_max),
                "--output",
                args.expert_segments,
                "--fcd-output-dir",
                "runs/fcd_expert_glosa",
            ]
        )
    maybe_run(
        [
            sys.executable,
            "train_sumo_rlhf.py",
            "--sumo-cfg",
            args.sumo_cfg,
            "--ego-id",
            "ego",
            "--episodes",
            str(args.episodes),
            "--step-length",
            str(args.step_length),
            "--segment-length",
            str(args.segment_length),
            "--seed",
            str(args.rl_seed),
            "--ego-depart-min",
            str(args.ego_depart_min),
            "--ego-depart-max",
            str(args.ego_depart_max),
            "--output",
            args.random_segments,
            "--fcd-output-dir",
            "runs/fcd_random",
        ],
        [args.random_segments],
        args.skip_existing,
    )
    run(
        [
            sys.executable,
            "merge_trajectory_segments.py",
            "--inputs",
            args.random_segments,
            args.expert_segments,
            "--output",
            args.preference_pool,
        ]
    )


def label(args, preference_pool: str | None = None, preferences: str | None = None, plot_dir: str | None = None):
    run(
        [
            sys.executable,
            "preference_web.py",
            "--segments",
            preference_pool or args.preference_pool,
            "--output",
            preferences or args.preferences,
            "--plot-dir",
            plot_dir or args.plot_dir,
            "--pairs",
            str(args.pairs),
            "--match-mode",
            args.match_mode,
            "--exit-when-done",
        ]
    )


def reward(args, preference_pool: str | None = None, preferences: str | None = None, reward_model: str | None = None):
    run(
        [
            sys.executable,
            "train_reward_model.py",
            "--segments",
            preference_pool or args.preference_pool,
            "--preferences",
            preferences or args.preferences,
            "--output",
            reward_model or args.reward_model,
        ]
    )


def policy(args, reward_model: str | None = None, policy_segments: str | None = None):
    run(
        [
            sys.executable,
            "train_sumo_rlhf.py",
            "--sumo-cfg",
            args.sumo_cfg,
            "--ego-id",
            "ego",
            "--episodes",
            str(args.policy_episodes),
            "--step-length",
            str(args.step_length),
            "--segment-length",
            str(args.segment_length),
            "--seed",
            str(args.policy_seed),
            "--ego-depart-min",
            str(args.ego_depart_min),
            "--ego-depart-max",
            str(args.ego_depart_max),
            "--reward-checkpoint",
            reward_model or args.reward_model,
            "--output",
            policy_segments or args.policy_segments,
            "--fcd-output-dir",
            "runs/fcd_rlhf",
        ]
    )


def round2(args, preference_pool: str | None = None, policy_segments: str | None = None, output_pool: str | None = None):
    run(
        [
            sys.executable,
            "merge_trajectory_segments.py",
            "--inputs",
            preference_pool or args.preference_pool,
            policy_segments or args.policy_segments,
            "--output",
            output_pool or args.round2_pool,
        ]
    )


def iterate(args):
    setup(args)
    collect(args)

    current_pool = args.preference_pool
    for round_idx in range(1, args.rounds + 1):
        preferences = f"runs/preferences_round{round_idx}.jsonl"
        reward_model = f"runs/reward_model_round{round_idx}.pt"
        policy_segments = f"runs/rlhf_policy_segments_round{round_idx}.jsonl"
        plot_dir = f"runs/preference_web_plots_round{round_idx}"
        next_pool = f"runs/preference_pool_round{round_idx + 1}.jsonl"

        print(f"\n=== RLHF round {round_idx}/{args.rounds} ===", flush=True)
        label(args, preference_pool=current_pool, preferences=preferences, plot_dir=plot_dir)
        reward(
            args,
            preference_pool=current_pool,
            preferences=preferences,
            reward_model=reward_model,
        )
        policy(args, reward_model=reward_model, policy_segments=policy_segments)

        if round_idx < args.rounds:
            round2(
                args,
                preference_pool=current_pool,
                policy_segments=policy_segments,
                output_pool=next_pool,
            )
            current_pool = next_pool


def main():
    args = parse_args()
    if args.stage == "setup":
        setup(args)
    elif args.stage == "collect":
        collect(args)
    elif args.stage == "label":
        label(args)
    elif args.stage == "reward":
        reward(args)
    elif args.stage == "policy":
        policy(args)
    elif args.stage == "round2":
        round2(args)
    elif args.stage == "iterate":
        iterate(args)
    elif args.stage == "all-before-label":
        setup(args)
        collect(args)
        print(
            "\nNext: run `python run_pipeline.py --stage label` "
            "and complete the web labels."
        )
    elif args.stage == "all-after-label":
        reward(args)
        policy(args)
        round2(args)


if __name__ == "__main__":
    main()
