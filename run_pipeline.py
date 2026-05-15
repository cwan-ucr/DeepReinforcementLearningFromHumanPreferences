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
            "pretrain",
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
    parser.add_argument(
        "--policy-rollouts-per-episode",
        type=int,
        default=2,
        help=(
            "Number of stochastic RL-policy rollouts to collect for each scenario "
            "episode during policy stages."
        ),
    )
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--pretrain-pairs", type=int, default=300)
    parser.add_argument("--step-length", type=float, default=0.5)
    parser.add_argument("--segment-length", type=int, default=20)
    parser.add_argument(
        "--whole-episode-segments",
        action="store_true",
        help="Use each full episode as one preference segment instead of fixed-length slices.",
    )
    parser.add_argument(
        "--animation-window-seconds",
        type=float,
        default=10.0,
        help="Preference animation window. Use <=0 for full trajectory.",
    )
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
    parser.add_argument("--pretrain-preferences", default="runs/pretrain_preferences.jsonl")
    parser.add_argument("--reward-model", default="runs/reward_model.pt")
    parser.add_argument("--pretrain-reward-model", default="runs/reward_model_pretrain.pt")
    parser.add_argument("--policy-segments", default="runs/rlhf_policy_segments.jsonl")
    parser.add_argument("--policy-checkpoint", default="runs/ppo_policy.pt")
    parser.add_argument("--round2-pool", default="runs/preference_pool_round2.jsonl")
    parser.add_argument("--plot-dir", default="runs/preference_web_plots")
    parser.add_argument("--match-mode", choices=["episode", "scene", "time", "position", "both", "random"], default="scene")
    parser.add_argument(
        "--human-source-pair-weights",
        default=(
            "rl-policy:rl-policy:0.65,"
            "rl-policy:glosa:0.12,"
            "rl-policy:sumo-default:0.10,"
            "rl-policy:rl-random:0.08,"
            "glosa:sumo-default:0.03,"
            "sumo-default:rl-random:0.02"
        ),
        help="Weighted source-pair schedule used by human labeling after pretraining.",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--reward-dropout", type=float, default=0.1)
    parser.add_argument("--reward-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def exists(path: str) -> bool:
    return Path(path).exists()


def episode_aligned_seed(args, source_seed: int) -> int:
    if args.match_mode == "episode":
        return args.seed
    return source_seed


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
        command = [
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
        if args.whole_episode_segments:
            command.append("--whole-episode-segments")
        run(command)
        command = [
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
        if args.whole_episode_segments:
            command.append("--whole-episode-segments")
        run(command)
    command = [
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
                str(episode_aligned_seed(args, args.rl_seed)),
            "--ego-depart-min",
            str(args.ego_depart_min),
            "--ego-depart-max",
            str(args.ego_depart_max),
            "--output",
            args.random_segments,
            "--fcd-output-dir",
            "runs/fcd_random",
    ]
    if args.whole_episode_segments:
        command.append("--whole-episode-segments")
    maybe_run(
        command,
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
    command = [
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
            "--animation-window-seconds",
            str(0.0 if args.whole_episode_segments else args.animation_window_seconds),
            "--exit-when-done",
    ]
    if args.human_source_pair_weights:
        command.extend(["--source-pair-weights", args.human_source_pair_weights])
    run(command)


def reward(
    args,
    preference_pool: str | None = None,
    preferences: str | list[str] | None = None,
    reward_model: str | None = None,
):
    preference_files = preferences or args.preferences
    if isinstance(preference_files, str):
        preference_files = [preference_files]
    command = [
            sys.executable,
            "train_reward_model.py",
            "--segments",
            preference_pool or args.preference_pool,
            "--preferences",
            *preference_files,
            "--output",
            reward_model or args.reward_model,
            "--ensemble-size",
            str(args.ensemble_size),
            "--dropout",
            str(args.reward_dropout),
            "--weight-decay",
            str(args.reward_weight_decay),
            "--skip-missing-preferences",
    ]
    run(command)


def pretrain(args):
    run(
        [
            sys.executable,
            "generate_preference_labels.py",
            "--segments",
            args.preference_pool,
            "--output",
            args.pretrain_preferences,
            "--pairs",
            str(args.pretrain_pairs),
            "--ranking",
            "glosa",
            "sumo-default",
            "rl-random",
            "--match-mode",
            args.match_mode,
            "--seed",
            str(args.seed),
            "--overwrite",
        ]
    )
    reward(
        args,
        preference_pool=args.preference_pool,
        preferences=args.pretrain_preferences,
        reward_model=args.pretrain_reward_model,
    )


def policy(
    args,
    reward_model: str | None = None,
    policy_segments: str | None = None,
    policy_checkpoint_in: str | None = None,
    policy_checkpoint_out: str | None = None,
):
    command = [
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
            str(episode_aligned_seed(args, args.policy_seed)),
            "--rollouts-per-episode",
            str(args.policy_rollouts_per_episode),
            "--ego-depart-min",
            str(args.ego_depart_min),
            "--ego-depart-max",
            str(args.ego_depart_max),
            "--reward-checkpoint",
            reward_model or args.reward_model,
            "--ppo-learning-rate",
            str(args.ppo_learning_rate),
            "--ppo-gamma",
            str(args.ppo_gamma),
            "--ppo-gae-lambda",
            str(args.ppo_gae_lambda),
            "--ppo-clip-ratio",
            str(args.ppo_clip_ratio),
            "--ppo-update-epochs",
            str(args.ppo_update_epochs),
            "--ppo-batch-size",
            str(args.ppo_batch_size),
            "--ppo-entropy-coef",
            str(args.ppo_entropy_coef),
            "--ppo-value-coef",
            str(args.ppo_value_coef),
            "--output",
            policy_segments or args.policy_segments,
            "--fcd-output-dir",
            "runs/fcd_rlhf",
    ]
    if args.whole_episode_segments:
        command.append("--whole-episode-segments")
    if policy_checkpoint_in:
        command.extend(["--policy-checkpoint-in", policy_checkpoint_in])
    command.extend(["--policy-checkpoint-out", policy_checkpoint_out or args.policy_checkpoint])
    run(command)


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
    pretrain(args)

    bootstrap_policy_segments = "runs/rlhf_policy_segments_round0.jsonl"
    bootstrap_policy_checkpoint = "runs/ppo_policy_round0.pt"
    bootstrap_pool = "runs/preference_pool_round1.jsonl"
    policy(
        args,
        reward_model=args.pretrain_reward_model,
        policy_segments=bootstrap_policy_segments,
        policy_checkpoint_out=bootstrap_policy_checkpoint,
    )
    round2(
        args,
        preference_pool=args.preference_pool,
        policy_segments=bootstrap_policy_segments,
        output_pool=bootstrap_pool,
    )

    current_pool = bootstrap_pool
    current_policy_checkpoint = bootstrap_policy_checkpoint
    for round_idx in range(1, args.rounds + 1):
        preferences = f"runs/preferences_round{round_idx}.jsonl"
        reward_model = f"runs/reward_model_round{round_idx}.pt"
        policy_segments = f"runs/rlhf_policy_segments_round{round_idx}.jsonl"
        policy_checkpoint = f"runs/ppo_policy_round{round_idx}.pt"
        plot_dir = f"runs/preference_web_plots_round{round_idx}"
        next_pool = f"runs/preference_pool_round{round_idx + 1}.jsonl"

        print(f"\n=== RLHF round {round_idx}/{args.rounds} ===", flush=True)
        label(args, preference_pool=current_pool, preferences=preferences, plot_dir=plot_dir)
        reward(
            args,
            preference_pool=current_pool,
            preferences=[args.pretrain_preferences, preferences],
            reward_model=reward_model,
        )
        policy(
            args,
            reward_model=reward_model,
            policy_segments=policy_segments,
            policy_checkpoint_in=current_policy_checkpoint,
            policy_checkpoint_out=policy_checkpoint,
        )
        current_policy_checkpoint = policy_checkpoint

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
    elif args.stage == "pretrain":
        pretrain(args)
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
