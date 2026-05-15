from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import random
import numpy as np

from sumo_rlhf.fcd import TrafficTrajectories, slice_traffic_trajectories


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
    traffic_trajectories: TrafficTrajectories = field(default_factory=dict)

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
        segment_prefix: Optional[str] = None,
        traffic_trajectories: Optional[TrafficTrajectories] = None,
        whole_episode: bool = False,
    ):
        if whole_episode:
            if not episode_steps:
                return
            base_id = f"ep{episode_id:05d}_full"
            segment_id = f"{segment_prefix}_{base_id}" if segment_prefix else base_id
            start_time = float(episode_steps[0].info.get("simulation_time", 0.0))
            end_time = float(episode_steps[-1].info.get("simulation_time", start_time))
            segment_traffic = (
                slice_traffic_trajectories(traffic_trajectories, start_time, end_time)
                if traffic_trajectories
                else {}
            )
            self.segments.append(
                TrajectorySegment(
                    segment_id,
                    episode_id,
                    episode_steps,
                    traffic_trajectories=segment_traffic,
                )
            )
            return

        if stride is None:
            stride = segment_length
        for start in range(0, max(len(episode_steps) - segment_length + 1, 0), stride):
            segment_steps = episode_steps[start : start + segment_length]
            base_id = f"ep{episode_id:05d}_t{start:05d}"
            segment_id = f"{segment_prefix}_{base_id}" if segment_prefix else base_id
            segment_traffic = {}
            if traffic_trajectories and segment_steps:
                start_time = float(segment_steps[0].info.get("simulation_time", start))
                end_time = float(
                    segment_steps[-1].info.get(
                        "simulation_time", start + segment_length - 1
                    )
                )
                segment_traffic = slice_traffic_trajectories(
                    traffic_trajectories, start_time, end_time
                )
            self.segments.append(
                TrajectorySegment(
                    segment_id,
                    episode_id,
                    segment_steps,
                    traffic_trajectories=segment_traffic,
                )
            )

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
                f.write(json.dumps(asdict(segment), default=json_numpy_default) + "\n")

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
                        traffic_trajectories=item.get("traffic_trajectories", {}),
                    )
                )
        return buffer


def iter_step_features(segment: TrajectorySegment) -> Iterable[tuple[List[float], float]]:
    for step in segment.steps:
        yield step.obs, step.action_value


def json_numpy_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
