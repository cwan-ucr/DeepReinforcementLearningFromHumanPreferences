from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from sumo_rlhf.reward_model import StepRewardModel
from sumo_rlhf.trajectory_buffer import TrajectorySegment


TTC_THRESHOLD_S = 3.0
STOP_SPEED_MPS = 0.2
VEHICLE_MASS_KG = 1500.0
ROLLING_RESISTANCE_COEFF = 0.01
AIR_DENSITY_KGPM3 = 1.225
DRAG_AREA_M2 = 0.6
GRAVITY_MPS2 = 9.81


@dataclass
class SegmentMetrics:
    segment_id: str
    episode_id: int
    source: str
    duration_s: float
    start_time_s: float
    end_time_s: float
    start_position_m: float
    end_position_m: float
    distance_m: float
    mean_speed_mps: float
    min_speed_mps: float
    max_speed_mps: float
    stop_count: int
    max_accel_mps2: float
    max_decel_mps2: float
    mean_abs_accel_mps2: float
    mean_abs_jerk_mps3: float
    max_abs_jerk_mps3: float
    energy_kwh_per_100km: float
    cumulative_tet_s: float
    cumulative_abs_jerk_mps2: float
    stop_time_s: float
    min_front_gap_m: float
    mean_front_gap_m: float
    min_time_headway_s: float
    mean_time_headway_s: float
    red_light_stop_distance_m: float
    reward_model_score: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def segment_source(segment: TrajectorySegment) -> str:
    if not segment.steps:
        return "unknown"
    return str(segment.steps[0].info.get("source", segment.segment_id.split("_", 1)[0]))


def compute_segment_metrics(
    segment: TrajectorySegment,
    reward_model: Optional[StepRewardModel] = None,
) -> SegmentMetrics:
    times = np.asarray(
        [step.info.get("simulation_time", np.nan) for step in segment.steps],
        dtype=np.float32,
    )
    raw_obs = [step.info.get("raw_observation", {}) for step in segment.steps]
    positions = np.asarray([raw.get("position", np.nan) for raw in raw_obs], dtype=np.float32)
    speeds = np.asarray([raw.get("speed", np.nan) for raw in raw_obs], dtype=np.float32)
    front_gaps = np.asarray(
        [raw.get("front_distance", np.nan) for raw in raw_obs], dtype=np.float32
    )
    delta_v_front = np.asarray(
        [raw.get("delta_v_front", np.nan) for raw in raw_obs], dtype=np.float32
    )
    energy_rate_kj_s = np.asarray(
        [step.info.get("energy_consumption_kj_s", np.nan) for step in segment.steps],
        dtype=np.float32,
    )
    actions = np.asarray([step.action_value for step in segment.steps], dtype=np.float32)

    duration = _duration(times)
    distance = _finite_last(positions) - _finite_first(positions)
    actual_accel = _actual_accel(times, speeds)
    jerk = _jerk(times, actual_accel)
    moving = speeds > 0.1
    time_headway = np.where(moving, front_gaps / np.maximum(speeds, 0.1), np.nan)
    cumulative_energy_kj = _cumulative_recorded_energy_kj(times, energy_rate_kj_s)
    if not np.isfinite(cumulative_energy_kj):
        cumulative_energy_kj = _cumulative_energy_kj(times, speeds, actual_accel)
    energy_kwh_per_100km = _energy_kwh_per_100km(cumulative_energy_kj, distance)
    cumulative_tet = _cumulative_tet(times, front_gaps, delta_v_front)
    cumulative_abs_jerk = _integrate_abs(times, jerk)
    stop_time = _stop_time(times, speeds)

    reward_score = None
    if reward_model is not None and segment.steps:
        reward_score = float(
            reward_model.segment_return(
                [(step.obs, float(step.action_value)) for step in segment.steps]
            ).item()
        )

    return SegmentMetrics(
        segment_id=segment.segment_id,
        episode_id=int(segment.episode_id),
        source=segment_source(segment),
        duration_s=duration,
        start_time_s=_finite_first(times),
        end_time_s=_finite_last(times),
        start_position_m=_finite_first(positions),
        end_position_m=_finite_last(positions),
        distance_m=distance,
        mean_speed_mps=_nanmean(speeds),
        min_speed_mps=_nanmin(speeds),
        max_speed_mps=_nanmax(speeds),
        stop_count=_stop_count(speeds),
        max_accel_mps2=_nanmax(actual_accel),
        max_decel_mps2=_nanmin(actual_accel),
        mean_abs_accel_mps2=_nanmean(np.abs(actions)),
        mean_abs_jerk_mps3=_nanmean(np.abs(jerk)),
        max_abs_jerk_mps3=_nanmax(np.abs(jerk)),
        energy_kwh_per_100km=energy_kwh_per_100km,
        cumulative_tet_s=cumulative_tet,
        cumulative_abs_jerk_mps2=cumulative_abs_jerk,
        stop_time_s=stop_time,
        min_front_gap_m=_nanmin(front_gaps),
        mean_front_gap_m=_nanmean(front_gaps),
        min_time_headway_s=_nanmin(time_headway),
        mean_time_headway_s=_nanmean(time_headway),
        red_light_stop_distance_m=_red_light_stop_distance(raw_obs, speeds),
        reward_model_score=reward_score,
    )


def aggregate_metrics(metrics: Iterable[SegmentMetrics]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[SegmentMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.source].append(metric)

    rows = []
    for source, items in sorted(grouped.items()):
        row: Dict[str, object] = {"source": source, "count": len(items)}
        keys = [
            "duration_s",
            "distance_m",
            "mean_speed_mps",
            "stop_count",
            "max_accel_mps2",
            "max_decel_mps2",
            "mean_abs_accel_mps2",
            "mean_abs_jerk_mps3",
            "max_abs_jerk_mps3",
            "energy_kwh_per_100km",
            "cumulative_tet_s",
            "cumulative_abs_jerk_mps2",
            "stop_time_s",
            "min_front_gap_m",
            "mean_front_gap_m",
            "min_time_headway_s",
            "mean_time_headway_s",
            "red_light_stop_distance_m",
            "reward_model_score",
        ]
        for key in keys:
            values = [getattr(item, key) for item in items]
            values = [value for value in values if value is not None]
            row[key] = _nanmean(np.asarray(values, dtype=np.float32))
        rows.append(row)
    return rows


def compact_metric_cards(segment: TrajectorySegment) -> List[Dict[str, str]]:
    metrics = compute_segment_metrics(segment)
    return [
        {"label": "mean speed", "value": f"{metrics.mean_speed_mps:.1f} m/s"},
        {
            "label": "energy",
            "value": _format_optional(metrics.energy_kwh_per_100km, "kWh/100km"),
        },
        {"label": "TET", "value": f"{metrics.cumulative_tet_s:.1f} s"},
        {"label": "cum |jerk|", "value": f"{metrics.cumulative_abs_jerk_mps2:.1f} m/s2"},
        {"label": "stop time", "value": f"{metrics.stop_time_s:.1f} s"},
    ]


def _actual_accel(times: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    values = np.full_like(speeds, np.nan, dtype=np.float32)
    if len(speeds) < 2:
        return values
    dt = np.diff(times)
    dv = np.diff(speeds)
    valid = np.isfinite(dt) & np.isfinite(dv) & (dt > 1e-6)
    values[1:][valid] = dv[valid] / dt[valid]
    return values


def _jerk(times: np.ndarray, accel: np.ndarray) -> np.ndarray:
    values = np.full_like(accel, np.nan, dtype=np.float32)
    if len(accel) < 2:
        return values
    dt = np.diff(times)
    da = np.diff(accel)
    valid = np.isfinite(dt) & np.isfinite(da) & (dt > 1e-6)
    values[1:][valid] = da[valid] / dt[valid]
    return values


def _red_light_stop_distance(raw_obs: List[dict], speeds: np.ndarray) -> float:
    candidates = []
    for raw, speed in zip(raw_obs, speeds):
        if (
            float(raw.get("tls_red", 0.0)) > 0.5
            and np.isfinite(speed)
            and speed < 0.2
            and raw.get("tls_distance") is not None
        ):
            candidates.append(float(raw["tls_distance"]))
    if not candidates:
        return float("nan")
    return float(min(candidates))


def _stop_count(speeds: np.ndarray) -> int:
    stopped = np.isfinite(speeds) & (speeds < 0.2)
    count = 0
    was_stopped = False
    for is_stopped in stopped:
        if bool(is_stopped) and not was_stopped:
            count += 1
        was_stopped = bool(is_stopped)
    return count


def _step_durations(times: np.ndarray) -> np.ndarray:
    durations = np.zeros_like(times, dtype=np.float32)
    if len(times) < 2:
        return durations
    dt = np.diff(times)
    valid = np.isfinite(dt) & (dt > 1e-6)
    durations[:-1][valid] = dt[valid]
    return durations


def _cumulative_energy_kj(
    times: np.ndarray,
    speeds: np.ndarray,
    actual_accel: np.ndarray,
) -> float:
    dt = _step_durations(times)
    speed = np.where(np.isfinite(speeds), speeds, 0.0)
    accel = np.where(np.isfinite(actual_accel), actual_accel, 0.0)
    force = (
        VEHICLE_MASS_KG * accel
        + ROLLING_RESISTANCE_COEFF * VEHICLE_MASS_KG * GRAVITY_MPS2
        + 0.5 * AIR_DENSITY_KGPM3 * DRAG_AREA_M2 * speed * speed
    )
    traction_power = np.maximum(force * speed, 0.0)
    return float(np.sum(traction_power * dt) / 1000.0)


def _cumulative_recorded_energy_kj(
    times: np.ndarray,
    energy_rate_kj_s: np.ndarray,
) -> float:
    if not np.any(np.isfinite(energy_rate_kj_s)):
        return float("nan")
    dt = _step_durations(times)
    rate = np.where(np.isfinite(energy_rate_kj_s), energy_rate_kj_s, 0.0)
    return float(np.sum(rate * dt))


def _energy_kwh_per_100km(energy_kj: float, distance_m: float) -> float:
    if not np.isfinite(energy_kj) or not np.isfinite(distance_m) or distance_m <= 1e-6:
        return float("nan")
    energy_kwh = energy_kj / 3600.0
    distance_km = distance_m / 1000.0
    return float(energy_kwh / distance_km * 100.0)


def _cumulative_tet(
    times: np.ndarray,
    front_gaps: np.ndarray,
    delta_v_front: np.ndarray,
    ttc_threshold_s: float = TTC_THRESHOLD_S,
) -> float:
    dt = _step_durations(times)
    closing_speed = delta_v_front
    ttc = np.full_like(front_gaps, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(front_gaps)
        & np.isfinite(closing_speed)
        & (front_gaps > 0.0)
        & (closing_speed > 0.1)
    )
    ttc[valid] = front_gaps[valid] / closing_speed[valid]
    exposed = np.isfinite(ttc) & (ttc < ttc_threshold_s)
    return float(np.sum(dt[exposed]))


def _integrate_abs(times: np.ndarray, values: np.ndarray) -> float:
    dt = _step_durations(times)
    finite_values = np.where(np.isfinite(values), np.abs(values), 0.0)
    return float(np.sum(finite_values * dt))


def _stop_time(times: np.ndarray, speeds: np.ndarray) -> float:
    dt = _step_durations(times)
    stopped = np.isfinite(speeds) & (speeds < STOP_SPEED_MPS)
    return float(np.sum(dt[stopped]))


def _duration(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return max(0.0, _finite_last(times) - _finite_first(times))


def _finite_first(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else float("nan")


def _finite_last(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _format_optional(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.1f} {unit}"
