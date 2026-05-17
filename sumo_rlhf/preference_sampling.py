from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np


SCENE_POSITION_MIN = 0.0
SCENE_POSITION_MAX = 350.0
SCENE_POSITION_BIN_SIZE = 50.0
SCENE_POSITION_BIN_COUNT = int((SCENE_POSITION_MAX - SCENE_POSITION_MIN) / SCENE_POSITION_BIN_SIZE)
SCENE_PHASE_BIN_COUNT = 4
SIGNAL_CYCLE_SECONDS = 90.0
CONTROL_SCENES = (
    "front0_pass1",
    "front0_pass0",
    "front1_pass1",
    "front1_pass0",
)


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


def segment_episode_id(segment) -> int:
    return int(segment.episode_id)


def segment_source(segment) -> str:
    if not segment.steps:
        return "unknown"
    return str(segment.steps[0].info.get("source", segment.segment_id.split("_", 1)[0]))


def _segment_decision_raw_observation(segment) -> dict:
    if not segment.steps:
        return {}
    candidate = None
    for step in segment.steps:
        raw = step.info.get("raw_observation", {})
        tls_distance = raw.get("tls_distance")
        if tls_distance is None:
            continue
        try:
            dist = float(tls_distance)
        except (TypeError, ValueError):
            continue
        if 20.0 <= dist <= 120.0:
            candidate = raw
            break
    if candidate is not None:
        return candidate
    return segment.steps[0].info.get("raw_observation", {})


def _infer_segment_control_scene(segment) -> str:
    raw = _segment_decision_raw_observation(segment)
    front_id = raw.get("front_vehicle_id")
    front_distance = raw.get("front_distance")
    has_front = False
    if front_id not in (None, "", "None"):
        has_front = True
    if front_distance is not None:
        try:
            if float(front_distance) < 80.0:
                has_front = True
        except (TypeError, ValueError):
            pass

    tls_green = float(raw.get("tls_green", 0.0) or 0.0) > 0.5
    speed = float(raw.get("speed", 0.0) or 0.0)
    tls_distance = float(raw.get("tls_distance", np.inf) or np.inf)
    tls_remaining = float(raw.get("tls_time_remaining", 0.0) or 0.0)
    eta_to_stopline = tls_distance / max(speed, 0.5)
    pass_now = bool(tls_green and eta_to_stopline <= max(tls_remaining - 1.0, 0.0))

    return f"front{1 if has_front else 0}_pass{1 if pass_now else 0}"


def load_manual_scene_labels(path: str | Path | None) -> dict[int, str]:
    """Load manual scene labels keyed by episode id.

    Supports:
    - JSON object: {"12": "front1_pass0", "13": "front0_pass1"}
    - JSONL rows: {"episode_id": 12, "scene": "front1_pass0"}
    """

    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"scene labels file not found: {file_path}")

    labels: dict[int, str] = {}
    if file_path.suffix.lower() == ".jsonl":
        for line in file_path.read_text(encoding="utf-8").splitlines():
            row_text = line.strip()
            if not row_text:
                continue
            row = json.loads(row_text)
            episode_id = int(row["episode_id"])
            scene = str(row["scene"])
            if scene in CONTROL_SCENES:
                labels[episode_id] = scene
        return labels

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene labels JSON must be an object keyed by episode_id.")
    for key, value in payload.items():
        scene = str(value)
        if scene not in CONTROL_SCENES:
            continue
        labels[int(key)] = scene
    return labels


def segment_control_scene(
    segment,
    manual_scene_labels: dict[int, str] | None = None,
    manual_scene_only: bool = False,
) -> str:
    if manual_scene_labels:
        scene = manual_scene_labels.get(segment_episode_id(segment))
        if scene in CONTROL_SCENES:
            return scene
        if manual_scene_only:
            return "unlabeled"
    return _infer_segment_control_scene(segment)


def parse_scenario_weights(spec: str | None) -> list[tuple[str, float]]:
    """Parse entries like 'front0_pass1:0.1,front0_pass0:0.4'."""

    if not spec:
        return []
    entries = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                "scenario weights must use 'scene_key:weight' entries."
            )
        scene_key, weight_text = parts
        if scene_key not in CONTROL_SCENES:
            raise ValueError(
                f"unknown scene key '{scene_key}'. Valid keys: {', '.join(CONTROL_SCENES)}"
            )
        weight = float(weight_text)
        if weight <= 0:
            continue
        entries.append((scene_key, weight))
    return entries


def load_scenario_weights(path: str | Path | None) -> list[tuple[str, float]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario weights file must contain a JSON object.")
    entries = []
    for scene_key, weight in payload.items():
        if scene_key not in CONTROL_SCENES:
            continue
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if numeric_weight > 0:
            entries.append((scene_key, numeric_weight))
    return entries


def _bin_index(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        return 0
    ratio = (float(value) - low) / (high - low)
    return max(0, min(bins - 1, int(ratio * bins)))


def _position_bin_index(position: float) -> int:
    relative_position = float(position) - SCENE_POSITION_MIN
    bin_index = int(relative_position // SCENE_POSITION_BIN_SIZE)
    return max(0, min(SCENE_POSITION_BIN_COUNT - 1, bin_index))


def segment_signal_phase(segment) -> float:
    return segment_start_time(segment) % SIGNAL_CYCLE_SECONDS


def segment_scene(segment) -> tuple[int, int]:
    """Return (50m position bin, signal phase bin)."""

    position_bin = _position_bin_index(segment_start_position(segment))
    phase_bin = _bin_index(
        segment_signal_phase(segment),
        0.0,
        SIGNAL_CYCLE_SECONDS,
        SCENE_PHASE_BIN_COUNT,
    )
    return position_bin, phase_bin


def format_segment_scene(segment) -> str:
    position_bin, phase_bin = segment_scene(segment)
    position_low = SCENE_POSITION_MIN + position_bin * SCENE_POSITION_BIN_SIZE
    position_high = position_low + SCENE_POSITION_BIN_SIZE
    return f"{position_low:.0f}-{position_high:.0f}m/phase{phase_bin + 1}"


def _matches(left, right, match_mode: str, position_tol: float, time_tol: float) -> bool:
    if match_mode == "random":
        return True
    if match_mode == "episode":
        return segment_episode_id(left) == segment_episode_id(right)
    if match_mode == "scene":
        return segment_scene(left) == segment_scene(right)

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
    if match_mode == "episode":
        return abs(segment_episode_id(right) - segment_episode_id(left))
    if match_mode == "scene":
        left_pos_bin, left_phase_bin = segment_scene(left)
        right_pos_bin, right_phase_bin = segment_scene(right)
        phase_delta = abs(right_phase_bin - left_phase_bin)
        phase_delta = min(phase_delta, SCENE_PHASE_BIN_COUNT - phase_delta)
        return abs(right_pos_bin - left_pos_bin) + phase_delta
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
    match_mode: str = "scene",
    prefer_different_source: bool = True,
):
    if match_mode == "episode":
        source_passes = [True, False] if prefer_different_source else [False]
        for require_different_source in source_passes:
            pair = _sample_same_episode_pair(
                segments,
                "*",
                "*",
                require_different_source=require_different_source,
            )
            if pair is not None:
                return (*pair, True)

    if match_mode == "scene":
        source_passes = [True, False] if prefer_different_source else [False]
        for require_different_source in source_passes:
            pair = _sample_same_scene_pair(
                segments,
                "*",
                "*",
                require_different_source=require_different_source,
            )
            if pair is not None:
                return (*pair, True)

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


def parse_source_pair_weights(spec: str | None) -> list[tuple[str, str, float]]:
    """Parse entries like 'rl-policy:*:0.8,glosa:sumo-default:0.2'."""

    if not spec:
        return []
    entries = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "source pair weights must use 'left_source:right_source:weight' entries."
            )
        left_source, right_source, weight_text = parts
        weight = float(weight_text)
        if weight <= 0:
            continue
        entries.append((left_source, right_source, weight))
    return entries


def _source_matches(pattern: str, source: str) -> bool:
    return pattern == "*" or pattern == source


def _segments_for_pair(segments, left_source: str, right_source: str):
    left_segments = [
        segment for segment in segments if _source_matches(left_source, segment_source(segment))
    ]
    right_segments = [
        segment
        for segment in segments
        if _source_matches(right_source, segment_source(segment))
    ]
    return left_segments, right_segments


def _sample_same_scene_pair(
    segments,
    left_source: str,
    right_source: str,
    require_different_source: bool = False,
):
    scene_keys = list({segment_scene(segment) for segment in segments})
    random.shuffle(scene_keys)
    valid_scenes = []
    for scene_key in scene_keys:
        left_segments = [
            segment
            for segment in segments
            if segment_scene(segment) == scene_key
            and _source_matches(left_source, segment_source(segment))
        ]
        right_segments = [
            segment
            for segment in segments
            if segment_scene(segment) == scene_key
            and _source_matches(right_source, segment_source(segment))
        ]
        has_valid_pair = any(
            right.segment_id != left.segment_id
            and (
                not require_different_source
                or segment_source(right) != segment_source(left)
            )
            for left in left_segments
            for right in right_segments
        )
        if has_valid_pair:
            valid_scenes.append((left_segments, right_segments))

    if not valid_scenes:
        return None

    left_segments, right_segments = random.choice(valid_scenes)
    pairs = [
        (left, right)
        for left in left_segments
        for right in right_segments
        if right.segment_id != left.segment_id
        and (
            not require_different_source
            or segment_source(right) != segment_source(left)
        )
    ]
    return random.choice(pairs)


def _sample_same_episode_pair(
    segments,
    left_source: str,
    right_source: str,
    require_different_source: bool = False,
):
    episode_ids = list({segment_episode_id(segment) for segment in segments})
    random.shuffle(episode_ids)
    valid_episodes = []
    for episode_id in episode_ids:
        left_segments = [
            segment
            for segment in segments
            if segment_episode_id(segment) == episode_id
            and _source_matches(left_source, segment_source(segment))
        ]
        right_segments = [
            segment
            for segment in segments
            if segment_episode_id(segment) == episode_id
            and _source_matches(right_source, segment_source(segment))
        ]
        has_valid_pair = any(
            right.segment_id != left.segment_id
            and (
                not require_different_source
                or segment_source(right) != segment_source(left)
            )
            for left in left_segments
            for right in right_segments
        )
        if has_valid_pair:
            valid_episodes.append((left_segments, right_segments))

    if not valid_episodes:
        return None

    left_segments, right_segments = random.choice(valid_episodes)
    pairs = [
        (left, right)
        for left in left_segments
        for right in right_segments
        if right.segment_id != left.segment_id
        and (
            not require_different_source
            or segment_source(right) != segment_source(left)
        )
    ]
    return random.choice(pairs)


def _randomize_pair_order(left, right):
    if random.random() < 0.5:
        return left, right
    return right, left


def sample_weighted_source_pair(
    segments,
    source_pair_weights: Sequence[tuple[str, str, float]],
    position_tol: float,
    time_tol: float,
    match_mode: str = "scene",
    scenario_weights: Sequence[tuple[str, float]] | None = None,
    manual_scene_labels: dict[int, str] | None = None,
    manual_scene_only: bool = False,
):
    if scenario_weights:
        segments_by_scene = {}
        for segment in segments:
            scene_key = segment_control_scene(
                segment,
                manual_scene_labels=manual_scene_labels,
                manual_scene_only=manual_scene_only,
            )
            if scene_key not in CONTROL_SCENES:
                continue
            segments_by_scene.setdefault(scene_key, []).append(segment)
        eligible_scenes = [
            (scene_key, weight)
            for scene_key, weight in scenario_weights
            if len(segments_by_scene.get(scene_key, [])) >= 2
        ]
        if eligible_scenes:
            scene_keys = [entry[0] for entry in eligible_scenes]
            scene_probs = [entry[1] for entry in eligible_scenes]
            chosen_scene = random.choices(scene_keys, weights=scene_probs, k=1)[0]
            scene_segments = segments_by_scene.get(chosen_scene, [])
            if len(scene_segments) >= 2:
                return sample_weighted_source_pair(
                    scene_segments,
                    source_pair_weights,
                    position_tol=position_tol,
                    time_tol=time_tol,
                    match_mode=match_mode,
                    scenario_weights=None,
                    manual_scene_labels=None,
                    manual_scene_only=False,
                )

    entries = []
    for entry in source_pair_weights:
        left_segments, right_segments = _segments_for_pair(segments, entry[0], entry[1])
        if left_segments and right_segments:
            entries.append(entry)
    if not entries:
        return sample_matched_pair(
            segments,
            position_tol=position_tol,
            time_tol=time_tol,
            match_mode=match_mode,
            prefer_different_source=True,
        )

    weights = [entry[2] for entry in entries]
    first_entry = random.choices(entries, weights=weights, k=1)[0]
    remaining_entries = [entry for entry in entries if entry != first_entry]
    random.shuffle(remaining_entries)
    for left_source, right_source, _weight in [first_entry, *remaining_entries]:
        if match_mode == "episode":
            pair = _sample_same_episode_pair(
                segments,
                left_source,
                right_source,
                require_different_source=False,
            )
            if pair is not None:
                return (*_randomize_pair_order(*pair), True)
            continue

        if match_mode == "scene":
            pair = _sample_same_scene_pair(
                segments,
                left_source,
                right_source,
                require_different_source=False,
            )
            if pair is not None:
                return (*_randomize_pair_order(*pair), True)

        left_segments, right_segments = _segments_for_pair(segments, left_source, right_source)
        anchors = left_segments[:]
        random.shuffle(anchors)
        for left in anchors:
            candidates = [
                right
                for right in right_segments
                if right.segment_id != left.segment_id
                and _matches(left, right, match_mode, position_tol, time_tol)
            ]
            if candidates:
                return (*_randomize_pair_order(left, random.choice(candidates)), True)

        pairs = [
            (left, right)
            for left in left_segments
            for right in right_segments
            if right.segment_id != left.segment_id
        ]
        if pairs:
            left, right = min(
                pairs,
                key=lambda pair: _match_distance(
                    pair[0],
                    pair[1],
                    match_mode,
                    position_tol,
                    time_tol,
                ),
            )
            return (*_randomize_pair_order(left, right), False)

    return sample_matched_pair(
        segments,
        position_tol=position_tol,
        time_tol=time_tol,
        match_mode=match_mode,
        prefer_different_source=True,
    )


def format_segment_summary(title, segment) -> str:
    summary = segment.summary()
    return (
        f"{title}: id={segment.segment_id} episode={segment.episode_id} "
        f"start_pos={segment_start_position(segment):.1f}m "
        f"start_time={segment_start_time(segment):.1f}s "
        f"scene={format_segment_scene(segment)} "
        f"source={segment_source(segment)} "
        f"length={summary.get('length', 0):.0f} "
        f"mean_speed={summary.get('mean_speed', 0):.2f} "
        f"min_front_distance={summary.get('min_front_distance', 0):.2f} "
        f"mean_abs_accel={summary.get('mean_abs_accel', 0):.2f}"
    )
