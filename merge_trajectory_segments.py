from __future__ import annotations

import argparse
from pathlib import Path

from sumo_rlhf.trajectory_buffer import TrajectoryBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge trajectory segment jsonl files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    merged = TrajectoryBuffer()
    seen = set()
    duplicate_count = 0

    for input_path in args.inputs:
        buffer = TrajectoryBuffer.load_jsonl(input_path)
        source_name = Path(input_path).stem
        for segment in buffer.segments:
            if segment.segment_id in seen:
                duplicate_count += 1
                segment.segment_id = f"{source_name}_{segment.segment_id}"
            seen.add(segment.segment_id)
            merged.segments.append(segment)

    merged.save_jsonl(args.output)
    print(
        f"merged {len(merged.segments)} segments into {args.output}; "
        f"renamed_duplicates={duplicate_count}"
    )


if __name__ == "__main__":
    main()

