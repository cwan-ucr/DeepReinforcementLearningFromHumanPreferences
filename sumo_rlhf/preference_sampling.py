from __future__ import annotations

import random
import re


def segment_start_position(segment) -> float:
    if not segment.steps:
        return 0.0
    raw = segment.steps[0].info.get("raw_observation", {})
    return float(raw.get("position", 0.0))


def segment_start_time(segment) -> float:
    if not segment.steps:
        return 0.0
    info = segment.steps[0].info
    if "simulation_time" in info:
        return float(info["simulation_time"])
    match = re.search(r"_t(\d+)$", segment.segment_id)
    if match:
        return float(match.group(1))
    return 0.0


def segment_source(segment) -> str:
    if not segment.steps:
        return "unknown"
    return str(segment.steps[0].info.get("source", segment.segment_id.split("_", 1)[0]))


def _matches(left, right, match_mode: str, position_tol: float, time_tol: float) -> bool:
    if match_mode == "random":
        return True

    position_ok = (
        abs(segment_start_position(right) - segment_start_position(left)) <= position_tol
    )
    time_ok = abs(segment_start_time(right) - segment_start_time(left)) <= time_tol

    if match_mode == "time":
        return time_ok
    if match_mode == "position":
        return position_ok
    if match_mode == "both":
        return position_ok and time_ok
    raise ValueError(f"Unknown match_mode: {match_mode}")


def _match_distance(left, right, match_mode: str, position_tol: float, time_tol: float) -> float:
    position_delta = abs(segment_start_position(right) - segment_start_position(left))
    time_delta = abs(segment_start_time(right) - segment_start_time(left))
    if match_mode == "time":
        return time_delta / max(time_tol, 1e-6)
    if match_mode == "position":
        return position_delta / max(position_tol, 1e-6)
    if match_mode == "both":
        return (
            position_delta / max(position_tol, 1e-6)
            + time_delta / max(time_tol, 1e-6)
        )
    return random.random()


def sample_matched_pair(
    segments,
    position_tol: float,
    time_tol: float,
    match_mode: str = "time",
    prefer_different_source: bool = True,
):
    anchors = segments[:]
    random.shuffle(anchors)

    source_passes = [True, False] if prefer_different_source else [False]
    for require_different_source in source_passes:
        for left in anchors:
            candidates = [
                right
                for right in segments
                if right.segment_id != left.segment_id
                and _matches(left, right, match_mode, position_tol, time_tol)
                and (
                    not require_different_source
                    or segment_source(right) != segment_source(left)
                )
            ]
            if candidates:
                return left, random.choice(candidates), True

    left, right = min(
        (
            (left, right)
            for left in segments
            for right in segments
            if right.segment_id != left.segment_id
        ),
        key=lambda pair: (
            0
            if (
                not prefer_different_source
                or segment_source(pair[0]) != segment_source(pair[1])
            )
            else 1,
            _match_distance(pair[0], pair[1], match_mode, position_tol, time_tol),
        ),
    )
    return left, right, False


def format_segment_summary(title, segment) -> str:
    summary = segment.summary()
    return (
        f"{title}: id={segment.segment_id} episode={segment.episode_id} "
        f"start_pos={segment_start_position(segment):.1f}m "
        f"start_time={segment_start_time(segment):.1f}s "
        f"source={segment_source(segment)} "
        f"length={summary.get('length', 0):.0f} "
        f"mean_speed={summary.get('mean_speed', 0):.2f} "
        f"min_front_distance={summary.get('min_front_distance', 0):.2f} "
        f"mean_abs_accel={summary.get('mean_abs_accel', 0):.2f}"
    )
