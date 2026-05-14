from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only preference labels whose segment ids exist in a segment pool."
    )
    parser.add_argument("--segments", default="runs/preference_pool.jsonl")
    parser.add_argument("--preferences", default="runs/preferences.jsonl")
    parser.add_argument("--output", default="runs/preferences_filtered.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    segment_ids = set()
    with Path(args.segments).open("r", encoding="utf-8") as f:
        for line in f:
            segment_ids.add(json.loads(line)["segment_id"])

    kept = 0
    dropped = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.preferences).open("r", encoding="utf-8") as fin, output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            pref = json.loads(line)
            if pref["left_id"] in segment_ids and pref["right_id"] in segment_ids:
                fout.write(json.dumps(pref) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"kept={kept} dropped={dropped} output={output}")


if __name__ == "__main__":
    main()
