from __future__ import annotations

import argparse
from pathlib import Path

from sumo_rlhf.preference_data import load_preference_examples
from sumo_rlhf.reward_model import (
    RewardEnsemble,
    StepRewardModel,
    save_reward_checkpoint,
    train_reward_ensemble,
    train_reward_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model from preferences.")
    parser.add_argument("--segments", default="runs/sumo_segments.jsonl")
    parser.add_argument(
        "--preferences",
        nargs="+",
        default=["runs/preferences.jsonl"],
        help="One or more preference jsonl files. Later human labels can be trained together with synthetic pretraining labels.",
    )
    parser.add_argument("--output", default="runs/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Train ensemble members on the same examples instead of bootstrap resamples.",
    )
    parser.add_argument(
        "--skip-missing-preferences",
        action="store_true",
        help="Skip labels that reference segment ids missing from the current segment pool.",
    )
    return parser.parse_args()


def infer_obs_dim(examples):
    if not examples or not examples[0].left:
        raise RuntimeError("No preference examples found.")
    return len(examples[0].left[0][0])


def main():
    args = parse_args()
    examples = []
    for preferences_path in args.preferences:
        examples.extend(
            load_preference_examples(
                args.segments,
                preferences_path,
                strict=not args.skip_missing_preferences,
            )
        )

    if args.ensemble_size > 1:
        model = RewardEnsemble(
            obs_dim=infer_obs_dim(examples),
            ensemble_size=args.ensemble_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        losses = train_reward_ensemble(
            model,
            examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bootstrap=not args.no_bootstrap,
        )
    else:
        model = StepRewardModel(
            obs_dim=infer_obs_dim(examples),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        losses = train_reward_model(
            model,
            examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    output = Path(args.output)
    save_reward_checkpoint(
        model,
        output,
        metadata={
            "preference_files": [str(path) for path in args.preferences],
            "examples": len(examples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )
    print(
        f"trained {args.ensemble_size} reward model(s) on {len(examples)} preference examples; "
        f"final_loss={losses[-1]:.4f}; saved {output}"
    )


if __name__ == "__main__":
    main()
