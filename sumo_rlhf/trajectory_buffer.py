from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import random


@dataclass
class StepRecord:
    obs: List[float]
    action_index: int
    action_value: float
    next_obs: List[float]
    done: bool
    info: Dict


@dataclass
class TrajectorySegment:
    segment_id: str
    episode_id: int
    steps: List[StepRecord]

    def summary(self) -> Dict[str, float]:
        if not self.steps:
            return {"length": 0.0}
        speeds = [s.info.get("raw_observation", {}).get("speed", 0.0) for s in self.steps]
        actions = [s.action_value for s in self.steps]
        return {
            "length": float(len(self.steps)),
            "mean_speed": float(sum(speeds) / max(len(speeds), 1)),
            "min_front_distance": float(
                min(
                    s.info.get("raw_observation", {}).get("front_distance", 0.0)
                    for s in self.steps
                )
            ),
            "mean_abs_accel": float(sum(abs(a) for a in actions) / max(len(actions), 1)),
        }


class TrajectoryBuffer:
    def __init__(self):
        self.segments: List[TrajectorySegment] = []

    def add_episode(
        self,
        episode_id: int,
        episode_steps: List[StepRecord],
        segment_length: int,
        stride: Optional[int] = None,
    ):
        if stride is None:
            stride = segment_length
        for start in range(0, max(len(episode_steps) - segment_length + 1, 0), stride):
            segment_steps = episode_steps[start : start + segment_length]
            segment_id = f"ep{episode_id:05d}_t{start:05d}"
            self.segments.append(TrajectorySegment(segment_id, episode_id, segment_steps))

    def sample_pair(self) -> Optional[tuple[TrajectorySegment, TrajectorySegment]]:
        if len(self.segments) < 2:
            return None
        a, b = random.sample(self.segments, 2)
        return a, b

    def save_jsonl(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for segment in self.segments:
                f.write(json.dumps(asdict(segment)) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "TrajectoryBuffer":
        buffer = cls()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                steps = [StepRecord(**step) for step in item["steps"]]
                buffer.segments.append(
                    TrajectorySegment(
                        segment_id=item["segment_id"],
                        episode_id=item["episode_id"],
                        steps=steps,
                    )
                )
        return buffer


def iter_step_features(segment: TrajectorySegment) -> Iterable[tuple[List[float], float]]:
    for step in segment.steps:
        yield step.obs, step.action_value

