from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET


TrafficTrajectories = Dict[str, List[dict]]


def load_fcd_trajectories(
    path: str | Path | None,
    ego_id: str = "ego",
) -> TrafficTrajectories:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    trajectories: TrafficTrajectories = {}
    root = ET.parse(path).getroot()
    for timestep in root.findall("timestep"):
        time_s = float(timestep.attrib["time"])
        for vehicle in timestep.findall("vehicle"):
            veh_id = vehicle.attrib.get("id")
            if not veh_id or veh_id == ego_id:
                continue
            # This generated scenario is a straight eastbound road, so FCD x is
            # the same longitudinal road position used in the trajectory plots.
            position = float(vehicle.attrib.get("x", "nan"))
            speed = float(vehicle.attrib.get("speed", "nan"))
            length = float(vehicle.attrib.get("length", "5.0"))
            trajectories.setdefault(veh_id, []).append(
                {
                    "time": time_s,
                    "position": position,
                    "speed": speed,
                    "length": length,
                }
            )
    return trajectories


def slice_traffic_trajectories(
    trajectories: TrafficTrajectories,
    start_time: float,
    end_time: float,
) -> TrafficTrajectories:
    sliced: TrafficTrajectories = {}
    for veh_id, points in trajectories.items():
        segment_points = [
            point for point in points if start_time <= float(point["time"]) <= end_time
        ]
        if segment_points:
            sliced[veh_id] = segment_points
    return sliced
