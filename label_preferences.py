from __future__ import annotations

import argparse
from pathlib import Path
import random

from sumo_rlhf.preference_data import PreferenceLabel, append_preference
from sumo_rlhf.preference_sampling import (
    format_segment_summary,
    parse_source_pair_weights,
    sample_matched_pair,
    sample_weighted_source_pair,
    segment_start_position,
    segment_start_time,
    segment_source,
)
from sumo_rlhf.trajectory_plot import plot_segment_pair
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Label pairwise trajectory preferences.")
    parser.add_argument("--segments", default="runs/sumo_segments.jsonl")
    parser.add_argument("--output", default="runs/preferences.jsonl")
    parser.add_argument("--plot-dir", default="runs/preference_plots")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--match-position-tol", type=float, default=30.0)
    parser.add_argument("--match-time-tol", type=float, default=20.0)
    parser.add_argument(
        "--match-mode",
        choices=["episode", "scene", "time", "position", "both", "random"],
        default="scene",
    )
    parser.add_argument("--allow-same-source", action="store_true")
    parser.add_argument("--source-pair-weights", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def format_summary(title, segment):
    return format_segment_summary(title, segment)


def main():
    args = parse_args()
    buffer = TrajectoryBuffer.load_jsonl(args.segments)
    if len(buffer.segments) < 2:
        raise RuntimeError("Need at least two trajectory segments to label preferences.")
    source_pair_weights = parse_source_pair_weights(args.source_pair_weights)

    for pair_idx in range(args.pairs):
        if source_pair_weights:
            left, right, matched = sample_weighted_source_pair(
                buffer.segments,
                source_pair_weights,
                position_tol=args.match_position_tol,
                time_tol=args.match_time_tol,
                match_mode=args.match_mode,
            )
        else:
            left, right, matched = sample_matched_pair(
                buffer.segments,
                position_tol=args.match_position_tol,
                time_tol=args.match_time_tol,
                match_mode=args.match_mode,
                prefer_different_source=not args.allow_same_source,
            )
        print("")
        print(format_summary("LEFT ", left))
        print(format_summary("RIGHT", right))
        pos_delta = abs(segment_start_position(left) - segment_start_position(right))
        time_delta = abs(segment_start_time(left) - segment_start_time(right))
        match_status = "matched" if matched else "closest fallback"
        print(
            f"pair match: {match_status}, mode={args.match_mode}, "
            f"source={segment_source(left)} vs {segment_source(right)}, "
            f"Δpos={pos_delta:.1f}m, Δtime={time_delta:.1f}s"
        )
        if not args.no_plots:
            plot_path = Path(args.plot_dir) / f"pair_{pair_idx:04d}_{left.segment_id}_vs_{right.segment_id}.png"
            plot_segment_pair(left, right, plot_path)
            print(f"plot: {plot_path}")
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
