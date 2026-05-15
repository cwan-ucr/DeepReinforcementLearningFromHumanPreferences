from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from sumo_rlhf.preference_data import PreferenceLabel
from sumo_rlhf.preference_sampling import (
    _match_distance,
    _matches,
    segment_source,
)
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic preference labels from a source ranking."
    )
    parser.add_argument("--segments", default="runs/preference_pool.jsonl")
    parser.add_argument("--output", default="runs/pretrain_preferences.jsonl")
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument(
        "--ranking",
        nargs="+",
        default=["glosa", "sumo-default", "rl-random"],
        help="Best-to-worst source ranking used for synthetic labels.",
    )
    parser.add_argument(
        "--adjacent-only",
        action="store_true",
        help="Only compare neighboring sources in the ranking.",
    )
    parser.add_argument("--match-mode", choices=["episode", "scene", "time", "position", "both", "random"], default="scene")
    parser.add_argument("--position-tol", type=float, default=30.0)
    parser.add_argument("--time-tol", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_pairs(ranking: list[str], adjacent_only: bool) -> list[tuple[str, str]]:
    if adjacent_only:
        return list(zip(ranking[:-1], ranking[1:]))
    pairs = []
    for i, better in enumerate(ranking):
        for worse in ranking[i + 1 :]:
            pairs.append((better, worse))
    return pairs


def choose_matched_pair(
    better_segments,
    worse_segments,
    *,
    match_mode: str,
    position_tol: float,
    time_tol: float,
):
    better = random.choice(better_segments)
    candidates = [
        segment
        for segment in worse_segments
        if _matches(better, segment, match_mode, position_tol, time_tol)
    ]
    if candidates:
        worse = random.choice(candidates)
    else:
        worse = min(
            worse_segments,
            key=lambda segment: _match_distance(
                better,
                segment,
                match_mode,
                position_tol,
                time_tol,
            ),
        )
    if random.random() < 0.5:
        return PreferenceLabel(
            left_id=better.segment_id,
            right_id=worse.segment_id,
            preference=0,
        )
    return PreferenceLabel(
        left_id=worse.segment_id,
        right_id=better.segment_id,
        preference=1,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")

    buffer = TrajectoryBuffer.load_jsonl(args.segments)
    by_source = {}
    for segment in buffer.segments:
        by_source.setdefault(segment_source(segment), []).append(segment)

    pairs = source_pairs(args.ranking, args.adjacent_only)
    missing_sources = [
        source for pair in pairs for source in pair if source not in by_source
    ]
    if missing_sources:
        available = ", ".join(sorted(by_source))
        raise KeyError(
            f"Missing source(s): {', '.join(sorted(set(missing_sources)))}. "
            f"Available sources: {available}"
        )

    labels = []
    for index in range(args.pairs):
        better_source, worse_source = pairs[index % len(pairs)]
        labels.append(
            choose_matched_pair(
                by_source[better_source],
                by_source[worse_source],
                match_mode=args.match_mode,
                position_tol=args.position_tol,
                time_tol=args.time_tol,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(asdict(label)) + "\n")

    print(
        f"wrote {len(labels)} synthetic preference labels to {output}; "
        f"ranking: {' > '.join(args.ranking)}"
    )


if __name__ == "__main__":
    main()
