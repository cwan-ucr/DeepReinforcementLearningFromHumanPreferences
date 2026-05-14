from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import json

from sumo_rlhf.reward_model import PreferenceExample
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer, iter_step_features


@dataclass
class PreferenceLabel:
    left_id: str
    right_id: str
    preference: int


def append_preference(path: str | Path, label: PreferenceLabel):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(label)) + "\n")


def load_preference_labels(path: str | Path) -> List[PreferenceLabel]:
    labels = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            labels.append(PreferenceLabel(**json.loads(line)))
    return labels


def load_preference_examples(
    segments_path: str | Path,
    preferences_path: str | Path,
) -> List[PreferenceExample]:
    buffer = TrajectoryBuffer.load_jsonl(segments_path)
    segment_by_id: Dict[str, object] = {
        segment.segment_id: segment for segment in buffer.segments
    }

    examples = []
    missing = []
    for label in load_preference_labels(preferences_path):
        left = segment_by_id.get(label.left_id)
        right = segment_by_id.get(label.right_id)
        if left is None or right is None:
            missing.append(label)
            continue
        examples.append(
            PreferenceExample(
                left=list(iter_step_features(left)),
                right=list(iter_step_features(right)),
                preference=label.preference,
            )
        )
    if missing:
        sample = ", ".join(
            f"({item.left_id}, {item.right_id})" for item in missing[:5]
        )
        raise KeyError(
            f"{len(missing)} preference labels reference segment ids that are not "
            f"in {segments_path}. First missing pairs: {sample}. "
            "Use the same preference_pool.jsonl that was used during labeling, "
            "or filter/relabel preferences after regenerating the pool."
        )
    return examples
