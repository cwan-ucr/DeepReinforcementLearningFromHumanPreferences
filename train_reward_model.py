from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sumo_rlhf.preference_data import load_preference_examples
from sumo_rlhf.reward_model import StepRewardModel, train_reward_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model from preferences.")
    parser.add_argument("--segments", default="runs/sumo_segments.jsonl")
    parser.add_argument("--preferences", default="runs/preferences.jsonl")
    parser.add_argument("--output", default="runs/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def infer_obs_dim(examples):
    if not examples or not examples[0].left:
        raise RuntimeError("No preference examples found.")
    return len(examples[0].left[0][0])


def main():
    args = parse_args()
    examples = load_preference_examples(args.segments, args.preferences)
    model = StepRewardModel(obs_dim=infer_obs_dim(examples))
    losses = train_reward_model(
        model,
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(
        f"trained on {len(examples)} preference examples; "
        f"final_loss={losses[-1]:.4f}; saved {output}"
    )


if __name__ == "__main__":
    main()

