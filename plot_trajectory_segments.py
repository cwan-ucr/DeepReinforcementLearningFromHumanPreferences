from __future__ import annotations

import argparse
import random
from pathlib import Path

from sumo_rlhf.trajectory_buffer import TrajectoryBuffer
from sumo_rlhf.trajectory_plot import (
    animate_segment,
    animate_segment_pair,
    plot_segment,
    plot_segment_pair,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stored SUMO trajectory segments.")
    parser.add_argument("--segments", default="runs/sumo_segments.jsonl")
    parser.add_argument("--output-dir", default="runs/trajectory_plots")
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--left-id", default=None)
    parser.add_argument("--right-id", default=None)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--animation-window-seconds", type=float, default=10.0)
    parser.add_argument("--animation-fps", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    buffer = TrajectoryBuffer.load_jsonl(args.segments)
    by_id = {segment.segment_id: segment for segment in buffer.segments}
    output_dir = Path(args.output_dir)

    if args.left_id or args.right_id:
        if not args.left_id or not args.right_id:
            raise RuntimeError("--left-id and --right-id must be provided together.")
        left = by_id[args.left_id]
        right = by_id[args.right_id]
        suffix = "gif" if args.animate else "png"
        output = output_dir / f"{left.segment_id}_vs_{right.segment_id}.{suffix}"
        if args.animate:
            animate_segment_pair(
                left,
                right,
                output,
                window_seconds=args.animation_window_seconds,
                fps=args.animation_fps,
            )
        else:
            plot_segment_pair(left, right, output)
        print(output)
        return

    if args.segment_id:
        segment = by_id[args.segment_id]
    else:
        segment = random.choice(buffer.segments)
    suffix = "gif" if args.animate else "png"
    output = output_dir / f"{segment.segment_id}.{suffix}"
    if args.animate:
        animate_segment(
            segment,
            output,
            window_seconds=args.animation_window_seconds,
            fps=args.animation_fps,
        )
    else:
        plot_segment(segment, output)
    print(output)


if __name__ == "__main__":
    main()
