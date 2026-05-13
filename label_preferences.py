from __future__ import annotations

import argparse
import random

from sumo_rlhf.preference_data import PreferenceLabel, append_preference
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Label pairwise trajectory preferences.")
    parser.add_argument("--segments", default="runs/sumo_segments.jsonl")
    parser.add_argument("--output", default="runs/preferences.jsonl")
    parser.add_argument("--pairs", type=int, default=20)
    return parser.parse_args()


def format_summary(title, segment):
    summary = segment.summary()
    return (
        f"{title}: id={segment.segment_id} episode={segment.episode_id} "
        f"length={summary.get('length', 0):.0f} "
        f"mean_speed={summary.get('mean_speed', 0):.2f} "
        f"min_front_distance={summary.get('min_front_distance', 0):.2f} "
        f"mean_abs_accel={summary.get('mean_abs_accel', 0):.2f}"
    )


def main():
    args = parse_args()
    buffer = TrajectoryBuffer.load_jsonl(args.segments)
    if len(buffer.segments) < 2:
        raise RuntimeError("Need at least two trajectory segments to label preferences.")

    for _ in range(args.pairs):
        left, right = random.sample(buffer.segments, 2)
        print("")
        print(format_summary("LEFT ", left))
        print(format_summary("RIGHT", right))
        answer = input("Preference: 1=left, 2=right, 3=neutral, q=quit > ").strip()
        if answer.lower() == "q":
            break
        if answer not in {"1", "2", "3"}:
            print("Skipped: invalid input.")
            continue

        preference = {"1": 0, "2": 1, "3": 2}[answer]
        append_preference(
            args.output,
            PreferenceLabel(
                left_id=left.segment_id,
                right_id=right.segment_id,
                preference=preference,
            ),
        )
        print(f"saved preference={preference}")


if __name__ == "__main__":
    main()

