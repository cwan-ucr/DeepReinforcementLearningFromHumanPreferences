from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sumo_rlhf.metrics import aggregate_metrics, compute_segment_metrics
from sumo_rlhf.reward_model import load_reward_checkpoint
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stored SUMO trajectory segments.")
    parser.add_argument("--segments", default="runs/preference_pool.jsonl")
    parser.add_argument("--output", default="runs/evaluation.csv")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--reward-checkpoint", default=None)
    return parser.parse_args()


def load_reward_model(buffer: TrajectoryBuffer, checkpoint: str | None):
    if checkpoint is None:
        return None
    if not buffer.segments or not buffer.segments[0].steps:
        raise RuntimeError("Cannot infer obs_dim from empty segment buffer.")
    obs_dim = len(buffer.segments[0].steps[0].obs)
    return load_reward_checkpoint(checkpoint, obs_dim=obs_dim)


def write_rows(path: str | Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    buffer = TrajectoryBuffer.load_jsonl(args.segments)
    reward_model = load_reward_model(buffer, args.reward_checkpoint)
    metrics = [
        compute_segment_metrics(segment, reward_model=reward_model)
        for segment in buffer.segments
    ]

    rows = [metric.as_dict() for metric in metrics]
    write_rows(args.output, rows)

    summary_output = args.summary_output
    if summary_output is None:
        output = Path(args.output)
        summary_output = output.with_name(output.stem + "_summary.csv")
    summary_rows = aggregate_metrics(metrics)
    write_rows(summary_output, summary_rows)

    print(
        f"evaluated {len(rows)} segments; wrote {args.output} and {summary_output}"
    )


if __name__ == "__main__":
    main()
