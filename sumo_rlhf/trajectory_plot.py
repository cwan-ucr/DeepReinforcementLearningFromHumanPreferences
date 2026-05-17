from __future__ import annotations

from pathlib import Path
from typing import List, Sequence
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from sumo_rlhf.preference_sampling import segment_source
from sumo_rlhf.trajectory_buffer import TrajectorySegment


TLS_COLORS = {
    "red": "#d62728",
    "yellow": "#ffbf00",
    "green": "#2ca02c",
}

DEFAULT_SIGNAL_CYCLE = 90.0
DEFAULT_GREEN_DURATION = 45.0
DEFAULT_VEHICLE_LENGTH = 5.0


def _raw_series(segment: TrajectorySegment, key: str, default: float = np.nan) -> List[float]:
    return [
        float(step.info.get("raw_observation", {}).get(key, default))
        for step in segment.steps
    ]


def _optional_raw_series(segment: TrajectorySegment, key: str) -> np.ndarray:
    values = []
    for step in segment.steps:
        raw = step.info.get("raw_observation", {})
        value = raw.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def _optional_info_series(segment: TrajectorySegment, key: str) -> np.ndarray:
    values = []
    for step in segment.steps:
        value = step.info.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def _time_axis(segment: TrajectorySegment) -> np.ndarray:
    times = [
        step.info.get("simulation_time")
        for step in segment.steps
        if step.info.get("simulation_time") is not None
    ]
    if len(times) == len(segment.steps) and times:
        return np.asarray(times, dtype=np.float32)

    match = re.search(r"_t(\d+)$", segment.segment_id)
    start_time = float(match.group(1)) if match else 0.0
    return start_time + np.arange(len(segment.steps), dtype=np.float32)


def _signal_state_at_time(time_s: float) -> str:
    cycle_time = time_s % DEFAULT_SIGNAL_CYCLE
    return "green" if cycle_time < DEFAULT_GREEN_DURATION else "red"


def _inferred_signal_positions(segment: TrajectorySegment) -> List[float]:
    positions = []
    for step in segment.steps:
        raw = step.info.get("raw_observation", {})
        state = str(raw.get("tls_state", "none"))
        if state == "none":
            continue
        if "tls_position" in raw:
            positions.append(float(raw["tls_position"]))
        elif "tls_distance" in raw and "position" in raw:
            positions.append(float(raw["position"]) + float(raw["tls_distance"]))

    unique_positions = []
    for position in positions:
        rounded = round(position / 10.0) * 10.0
        if not any(abs(rounded - existing) < 5.0 for existing in unique_positions):
            unique_positions.append(rounded)
    return unique_positions


def _signal_positions(segment: TrajectorySegment) -> List[float]:
    inferred_positions = _inferred_signal_positions(segment)
    return inferred_positions


def _padded_limits(
    values: Sequence[float],
    default: tuple[float, float],
    min_pad: float = 1.0,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    low = float(np.min(finite))
    high = float(np.max(finite))
    if abs(high - low) < 1e-6:
        low -= min_pad
        high += min_pad
    pad = max((high - low) * 0.08, min_pad)
    return low - pad, high + pad


def _segment_window(
    segment: TrajectorySegment,
    window_seconds: float | None = None,
) -> tuple[float, float]:
    t = _time_axis(segment)
    finite_t = t[np.isfinite(t)]
    start = float(finite_t[0]) if finite_t.size else 0.0
    if window_seconds is None:
        end = float(finite_t[-1]) if finite_t.size else start + 1.0
    else:
        end = start + float(window_seconds)
    return start, end


def _actual_accel_series(t: np.ndarray, speed: np.ndarray) -> np.ndarray:
    actual_accel = np.full_like(speed, np.nan, dtype=np.float32)
    if len(speed) < 2:
        return actual_accel
    dt = np.diff(t.astype(np.float32))
    dv = np.diff(speed.astype(np.float32))
    valid = np.isfinite(dt) & np.isfinite(dv) & (dt > 1e-6)
    actual_accel[1:][valid] = dv[valid] / dt[valid]
    return actual_accel


def _segment_y_limits(
    segment: TrajectorySegment,
    time_window: tuple[float, float],
) -> list[tuple[float, float]]:
    start, end = time_window
    t = _time_axis(segment)
    in_window = (t >= start) & (t <= end)
    position = np.asarray(_raw_series(segment, "position"), dtype=np.float32)
    speed = np.asarray(_raw_series(segment, "speed"), dtype=np.float32)
    safe_speed = _optional_info_series(segment, "safe_ref_speed")
    actions = np.asarray([step.action_value for step in segment.steps], dtype=np.float32)
    actual_accel = _actual_accel_series(t, speed)
    front_distance = np.asarray(
        _raw_series(segment, "front_distance"), dtype=np.float32
    )
    front_vehicle_position = _optional_raw_series(segment, "front_vehicle_position")
    rear_vehicle_position = _optional_raw_series(segment, "rear_vehicle_position")
    rear_distance = _optional_raw_series(segment, "rear_distance")
    signal_positions = _signal_positions(segment)

    position_values = list(position[in_window])
    position_values.extend(front_vehicle_position[in_window])
    position_values.extend(rear_vehicle_position[in_window])
    position_values.extend(signal_positions)
    for points in segment.traffic_trajectories.values():
        vehicle_t = np.asarray([p["time"] for p in points], dtype=np.float32)
        vehicle_position = np.asarray([p["position"] for p in points], dtype=np.float32)
        vehicle_mask = (vehicle_t >= start) & (vehicle_t <= end)
        position_values.extend(vehicle_position[vehicle_mask])

    speed_values = list(speed[in_window])
    speed_values.extend(safe_speed[in_window])
    gap_values = list(front_distance[in_window])
    gap_values.extend(rear_distance[in_window])
    return [
        _padded_limits(position_values, default=(0.0, 350.0), min_pad=5.0),
        _padded_limits(speed_values, default=(0.0, 15.0), min_pad=1.0),
        _padded_limits(
            list(actions[in_window]) + list(actual_accel[in_window]),
            default=(-3.5, 2.5),
            min_pad=0.5,
        ),
        _padded_limits(gap_values, default=(0.0, 120.0), min_pad=2.0),
    ]


def _plot_one_segment(
    axes: Sequence[plt.Axes],
    segment: TrajectorySegment,
    title: str,
    current_time: float | None = None,
    time_window: tuple[float, float] | None = None,
    y_limits: Sequence[tuple[float, float]] | None = None,
):
    t = _time_axis(segment)
    if time_window is not None:
        time_mask = (t >= time_window[0]) & (t <= time_window[1])
    else:
        time_mask = np.ones_like(t, dtype=bool)
    if current_time is not None:
        time_mask = time_mask & (t <= current_time)

    position = np.asarray(_raw_series(segment, "position"), dtype=np.float32)
    speed = np.asarray(_raw_series(segment, "speed"), dtype=np.float32)
    safe_speed = _optional_info_series(segment, "safe_ref_speed")
    front_distance = np.asarray(
        _raw_series(segment, "front_distance"), dtype=np.float32
    )
    actions = np.asarray([step.action_value for step in segment.steps], dtype=np.float32)
    front_vehicle_position = _optional_raw_series(segment, "front_vehicle_position")
    if np.all(np.isnan(front_vehicle_position)):
        front_vehicle_position = np.where(
            np.isfinite(position) & np.isfinite(front_distance) & (front_distance < 119.5),
            position + front_distance,
            np.nan,
        )
    rear_vehicle_position = _optional_raw_series(segment, "rear_vehicle_position")
    rear_distance = _optional_raw_series(segment, "rear_distance")
    rear_distance = np.where(np.isnan(rear_vehicle_position), np.nan, rear_distance)
    front_gap = np.where(np.isnan(front_vehicle_position), np.nan, front_distance)
    signal_positions = _signal_positions(segment)
    has_fcd_traffic = bool(segment.traffic_trajectories)

    if has_fcd_traffic:
        for idx, (veh_id, points) in enumerate(segment.traffic_trajectories.items()):
            vehicle_t = np.asarray([p["time"] for p in points], dtype=np.float32)
            vehicle_position = np.asarray(
                [p["position"] for p in points], dtype=np.float32
            )
            vehicle_mask = np.ones_like(vehicle_t, dtype=bool)
            if time_window is not None:
                vehicle_mask = (vehicle_t >= time_window[0]) & (vehicle_t <= time_window[1])
            if current_time is not None:
                vehicle_mask = vehicle_mask & (vehicle_t <= current_time)
            axes[0].plot(
                vehicle_t[vehicle_mask],
                vehicle_position[vehicle_mask],
                color="#9ca3af",
                linewidth=0.9,
                alpha=0.75,
                label="traffic vehicles" if idx == 0 else None,
            )
    axes[0].plot(t[time_mask], position[time_mask], color="#1f77b4", linewidth=2.2, label="ego")
    if not has_fcd_traffic:
        axes[0].plot(
            t[time_mask],
            front_vehicle_position[time_mask],
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.2,
            label="front vehicle",
        )
        axes[0].plot(
            t[time_mask],
            rear_vehicle_position[time_mask],
            color="#bcbd22",
            linestyle="--",
            linewidth=1.2,
            label="rear vehicle",
        )
    for idx, signal_position in enumerate(signal_positions):
        signal_mask = np.isfinite(t)
        if time_window is not None:
            signal_mask = signal_mask & (t >= time_window[0]) & (t <= time_window[1])
        if current_time is not None:
            signal_mask = signal_mask & (t <= current_time)
        signal_t = t[signal_mask]
        signal_colors = [
            TLS_COLORS[_signal_state_at_time(float(time_s))]
            for time_s in signal_t
        ]
        axes[0].scatter(
            signal_t,
            np.full_like(signal_t, signal_position, dtype=np.float32),
            c=signal_colors,
            s=22,
            marker="s",
            edgecolors="black",
            linewidths=0.2,
            label="traffic signal" if idx == 0 else None,
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("distance (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(t[time_mask], speed[time_mask], color="#17becf", linewidth=1.8)
    if np.any(np.isfinite(safe_speed[time_mask])):
        axes[1].plot(
            t[time_mask],
            safe_speed[time_mask],
            color="#f59e0b",
            linewidth=1.6,
            linestyle="--",
            label="safe speed",
        )
    axes[1].set_ylabel("speed (m/s)")
    axes[1].grid(True, alpha=0.25)
    if np.any(np.isfinite(safe_speed[time_mask])):
        axes[1].legend(loc="upper left", fontsize=8)

    axes[2].step(t[time_mask], actions[time_mask], where="post", color="#9467bd", linewidth=1.8)
    axes[2].set_ylabel("accel cmd")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t[time_mask], front_gap[time_mask], color="#8c564b", linewidth=1.8, label="front gap")
    axes[3].plot(t[time_mask], rear_distance[time_mask], color="#bcbd22", linewidth=1.8, label="rear gap")
    axes[3].set_ylabel("gap (m)")
    axes[3].set_xlabel("SUMO simulation time (s)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper left", fontsize=8)

    if current_time is not None:
        for axis in axes:
            axis.axvline(current_time, color="#111827", linewidth=1.0, alpha=0.55)
    if time_window is not None:
        for axis in axes:
            axis.set_xlim(time_window)
    if y_limits is not None:
        for axis, limits in zip(axes, y_limits):
            axis.set_ylim(limits)


def plot_segment_pair(
    left: TrajectorySegment,
    right: TrajectorySegment,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(14, 9),
        sharex="col",
        constrained_layout=True,
    )
    _plot_one_segment(
        axes[:, 0],
        left,
        f"LEFT: {left.segment_id} (episode {left.episode_id})",
    )
    _plot_one_segment(
        axes[:, 1],
        right,
        f"RIGHT: {right.segment_id} (episode {right.episode_id})",
    )
    fig.suptitle("Trajectory Segment Preference Comparison", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_segment(segment: TrajectorySegment, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    _plot_one_segment(axes, segment, f"{segment.segment_id} (episode {segment.episode_id})")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def segment_animation_payload(
    segment: TrajectorySegment,
    window_seconds: float = 10.0,
) -> dict:
    window = _segment_window(segment, window_seconds)
    t = _time_axis(segment)
    in_window = (t >= window[0]) & (t <= window[1])
    position = np.asarray(_raw_series(segment, "position"), dtype=np.float32)
    speed = np.asarray(_raw_series(segment, "speed"), dtype=np.float32)
    safe_speed = _optional_info_series(segment, "safe_ref_speed")
    actions = np.asarray([step.action_value for step in segment.steps], dtype=np.float32)
    actual_accel = _actual_accel_series(t, speed)
    front_distance = np.asarray(
        _raw_series(segment, "front_distance"), dtype=np.float32
    )
    front_vehicle_position = _optional_raw_series(segment, "front_vehicle_position")
    rear_vehicle_position = _optional_raw_series(segment, "rear_vehicle_position")
    rear_distance = _optional_raw_series(segment, "rear_distance")
    rear_distance = np.where(np.isnan(rear_vehicle_position), np.nan, rear_distance)
    front_gap = np.where(np.isnan(front_vehicle_position), np.nan, front_distance)

    traffic = []
    for vehicle_id, points in segment.traffic_trajectories.items():
        vehicle_t = np.asarray([p["time"] for p in points], dtype=np.float32)
        vehicle_position = np.asarray([p["position"] for p in points], dtype=np.float32)
        vehicle_mask = (vehicle_t >= window[0]) & (vehicle_t <= window[1])
        if np.any(vehicle_mask):
            traffic.append(
                {
                    "id": str(vehicle_id),
                    "time": vehicle_t[vehicle_mask].astype(float).tolist(),
                    "position": vehicle_position[vehicle_mask].astype(float).tolist(),
                    "length": float(points[0].get("length", DEFAULT_VEHICLE_LENGTH)),
                }
            )

    return {
        "id": segment.segment_id,
        "source": segment_source(segment),
        "episode": int(segment.episode_id),
        "window": [float(window[0]), float(window[1])],
        "time": t[in_window].astype(float).tolist(),
        "position": position[in_window].astype(float).tolist(),
        "speed": speed[in_window].astype(float).tolist(),
        "safeSpeed": safe_speed[in_window].astype(float).tolist(),
        "vehicleLength": DEFAULT_VEHICLE_LENGTH,
        "action": actions[in_window].astype(float).tolist(),
        "actualAccel": actual_accel[in_window].astype(float).tolist(),
        "frontGap": front_gap[in_window].astype(float).tolist(),
        "rearGap": rear_distance[in_window].astype(float).tolist(),
        "traffic": traffic,
        "signals": [float(value) for value in _signal_positions(segment)],
        "yLimits": [
            [float(low), float(high)]
            for low, high in _segment_y_limits(segment, window)
        ],
    }


def animate_segment_pair(
    left: TrajectorySegment,
    right: TrajectorySegment,
    output_path: str | Path,
    window_seconds: float = 10.0,
    fps: int = 4,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_window = _segment_window(left, window_seconds)
    right_window = _segment_window(right, window_seconds)
    left_limits = _segment_y_limits(left, left_window)
    right_limits = _segment_y_limits(right, right_window)
    frame_count = max(2, int(round(window_seconds * fps)) + 1)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(14, 9),
        sharex="col",
        constrained_layout=True,
    )

    def draw(frame_idx: int):
        elapsed = min(frame_idx / float(fps), window_seconds)
        for axis in axes.ravel():
            axis.clear()
        _plot_one_segment(
            axes[:, 0],
            left,
            f"LEFT: {left.segment_id} (episode {left.episode_id})",
            current_time=left_window[0] + elapsed,
            time_window=left_window,
            y_limits=left_limits,
        )
        _plot_one_segment(
            axes[:, 1],
            right,
            f"RIGHT: {right.segment_id} (episode {right.episode_id})",
            current_time=right_window[0] + elapsed,
            time_window=right_window,
            y_limits=right_limits,
        )
        fig.suptitle(
            f"Trajectory Segment Preference Animation: {elapsed:.1f}/{window_seconds:.1f}s",
            fontsize=14,
        )
        return list(axes.ravel())

    anim = animation.FuncAnimation(
        fig,
        draw,
        frames=frame_count,
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )
    try:
        anim.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    except Exception as exc:
        raise RuntimeError(
            "Failed to save trajectory animation. Install pillow if the GIF writer is missing."
        ) from exc
    finally:
        plt.close(fig)
    return output_path


def animate_segment(
    segment: TrajectorySegment,
    output_path: str | Path,
    window_seconds: float = 10.0,
    fps: int = 4,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segment_window = _segment_window(segment, window_seconds)
    y_limits = _segment_y_limits(segment, segment_window)
    frame_count = max(2, int(round(window_seconds * fps)) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True, constrained_layout=True)

    def draw(frame_idx: int):
        elapsed = min(frame_idx / float(fps), window_seconds)
        for axis in axes:
            axis.clear()
        _plot_one_segment(
            axes,
            segment,
            f"{segment.segment_id} (episode {segment.episode_id})",
            current_time=segment_window[0] + elapsed,
            time_window=segment_window,
            y_limits=y_limits,
        )
        fig.suptitle(f"Trajectory Segment Animation: {elapsed:.1f}/{window_seconds:.1f}s")
        return list(axes)

    anim = animation.FuncAnimation(
        fig,
        draw,
        frames=frame_count,
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )
    try:
        anim.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    except Exception as exc:
        raise RuntimeError(
            "Failed to save trajectory animation. Install pillow if the GIF writer is missing."
        ) from exc
    finally:
        plt.close(fig)
    return output_path
