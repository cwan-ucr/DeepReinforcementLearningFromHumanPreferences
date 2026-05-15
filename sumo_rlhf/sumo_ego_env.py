from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import os

FUEL_LOWER_HEATING_VALUE_KJ_PER_KG = 44000.0

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - import-time environment guard
    try:
        import gym
        from gym import spaces
    except ImportError as exc:
        raise ImportError(
            "sumo_rlhf expects either gymnasium or gym to be installed."
        ) from exc


@dataclass
class SumoEgoConfig:
    """Configuration for a single-lane arterial ego-vehicle SUMO environment."""

    sumo_cfg: str
    ego_id: str = "ego"
    gui: bool = False
    step_length: float = 1.0
    max_episode_steps: int = 300
    max_depart_delay_steps: int = 1000
    action_accels: Tuple[float, ...] = (
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    )
    max_position: float = 3000.0
    max_speed: float = 25.0
    max_front_dist: float = 120.0
    max_tl_dist: float = 500.0
    max_tl_time: float = 120.0
    normalize_observation: bool = True
    speed_mode: Optional[int] = None
    seed: Optional[int] = 42
    randomize_seed_on_reset: bool = True
    fcd_output_dir: Optional[str] = "runs/fcd"
    ego_route_id: str = "arterial_route"
    ego_type_id: Optional[str] = None
    ego_depart_min: float = 0.0
    ego_depart_max: float = 90.0
    ego_depart_speed: float = 0.0


class SumoEgoEnv(gym.Env):
    """Gym-style wrapper for longitudinal ego-vehicle control in SUMO.

    Observation layout:
        [position, speed, front_distance, ego_minus_front_speed,
         previous_action_accel, next_tls_distance, next_tls_time_remaining,
         tls_red, tls_yellow, tls_green]

    The environment returns a placeholder reward of 0.0. Training code should
    replace this with a learned reward model once preference data is available.
    """

    metadata = {"render.modes": []}

    def __init__(self, config: SumoEgoConfig):
        self.config = config
        self._traci = None
        self._sumolib = None
        self._steps = 0
        self._reset_count = 0
        self._active_seed = config.seed
        self._fcd_episode_index = 0
        self._current_fcd_path: Optional[Path] = None
        self._ego_depart_time = 0.0
        self._prev_action = 0.0
        self._last_obs = np.zeros(10, dtype=np.float32)

        self.action_space = spaces.Discrete(len(config.action_accels))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

    def reset(
        self,
        scenario_seed: Optional[int] = None,
        fcd_episode_index: Optional[int] = None,
    ):
        self.close()
        if scenario_seed is None:
            self._active_seed = self._next_seed()
        else:
            self._active_seed = int(scenario_seed)
            self._reset_count += 1
        self._fcd_episode_index = (
            max(self._reset_count - 1, 0)
            if fcd_episode_index is None
            else int(fcd_episode_index)
        )
        self._start_sumo()
        self._steps = 0
        self._prev_action = 0.0
        self._depart_ego()
        self._last_obs = self._get_observation()
        return self._last_obs.copy()

    def step(self, action: int):
        if self._traci is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        accel = float(self.config.action_accels[action])
        self._apply_acceleration(accel)
        self._traci.simulationStep()
        self._steps += 1
        self._prev_action = accel

        arrived = self.config.ego_id not in self._traci.vehicle.getIDList()
        timeout = self._steps >= self.config.max_episode_steps
        done = bool(arrived or timeout)

        if arrived:
            obs = np.zeros(10, dtype=np.float32)
        else:
            obs = self._get_observation()

        self._last_obs = obs.copy()
        info = {
            **self._current_info(accel, arrived),
            "timeout": timeout,
        }
        return obs, 0.0, done, info

    def passive_step(self):
        """Advance SUMO without applying an RL action and record actual acceleration."""
        if self._traci is None:
            raise RuntimeError("Call reset() before passive_step().")

        old_speed = 0.0
        if self.config.ego_id in self._traci.vehicle.getIDList():
            old_speed = float(self._traci.vehicle.getSpeed(self.config.ego_id))

        self._traci.simulationStep()
        self._steps += 1

        arrived = self.config.ego_id not in self._traci.vehicle.getIDList()
        timeout = self._steps >= self.config.max_episode_steps
        done = bool(arrived or timeout)

        if arrived:
            actual_accel = 0.0
            obs = np.zeros(10, dtype=np.float32)
        else:
            new_speed = float(self._traci.vehicle.getSpeed(self.config.ego_id))
            actual_accel = (new_speed - old_speed) / self.config.step_length
            self._prev_action = actual_accel
            obs = self._get_observation()

        self._last_obs = obs.copy()
        info = {
            **self._current_info(actual_accel, arrived),
            "timeout": timeout,
        }
        return obs, 0.0, done, info

    def observe(self) -> np.ndarray:
        if self._traci is None:
            raise RuntimeError("Call reset() before observe().")
        self._last_obs = self._get_observation()
        return self._last_obs.copy()

    def close(self):
        if self._traci is not None:
            try:
                self._traci.close(True)
            finally:
                self._traci = None

    @property
    def fcd_output_path(self) -> Optional[Path]:
        return self._current_fcd_path

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)
        self.config.seed = seed
        self._active_seed = seed

    def _start_sumo(self):
        try:
            import sumolib
            import traci
        except ImportError as exc:
            raise ImportError(
                "SUMO Python tools are required. Install SUMO and make sure "
                "traci/sumolib are importable, for example via SUMO_HOME/tools."
            ) from exc

        binary_name = "sumo-gui" if self.config.gui else "sumo"
        sumo_binary = sumolib.checkBinary(binary_name)
        cmd = [
            sumo_binary,
            "-c",
            self.config.sumo_cfg,
            "--step-length",
            str(self.config.step_length),
            "--no-warnings",
            "true",
        ]
        if self._active_seed is not None:
            cmd.extend(["--seed", str(int(self._active_seed))])
        self._current_fcd_path = self._make_fcd_output_path()
        if self._current_fcd_path is not None:
            self._current_fcd_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--fcd-output", str(self._current_fcd_path)])
        traci.start(cmd, port=self._free_port())
        self._traci = traci
        self._sumolib = sumolib

    @staticmethod
    def _free_port() -> int:
        return 20000 + (os.getpid() % 20000)

    def _next_seed(self) -> Optional[int]:
        reset_index = self._reset_count
        self._reset_count += 1
        if self.config.seed is None:
            return None
        if not self.config.randomize_seed_on_reset:
            return int(self.config.seed)
        return int(self.config.seed) + reset_index

    def _depart_ego(self):
        rng = np.random.default_rng(self._active_seed)
        depart_min = float(self.config.ego_depart_min)
        depart_max = float(self.config.ego_depart_max)
        if depart_max < depart_min:
            raise ValueError("ego_depart_max must be >= ego_depart_min.")
        self._ego_depart_time = float(rng.uniform(depart_min, depart_max))

        while float(self._traci.simulation.getTime()) < self._ego_depart_time:
            self._traci.simulationStep()

        if self.config.ego_id not in self._traci.vehicle.getIDList():
            self._traci.vehicle.add(
                vehID=self.config.ego_id,
                routeID=self.config.ego_route_id,
                typeID=self._ego_type_id(),
                depart="now",
                departLane="first",
                departPos="base",
                departSpeed=str(float(self.config.ego_depart_speed)),
            )

        for _ in range(self.config.max_depart_delay_steps):
            if self.config.ego_id in self._traci.vehicle.getIDList():
                break
            self._traci.simulationStep()
        else:
            raise RuntimeError(
                f"Ego vehicle '{self.config.ego_id}' could not be inserted after "
                f"random depart time {self._ego_depart_time:.2f}."
            )

        if self.config.speed_mode is not None:
            self._traci.vehicle.setSpeedMode(
                self.config.ego_id, int(self.config.speed_mode)
            )

    def _ego_type_id(self) -> str:
        if self.config.ego_type_id:
            return self.config.ego_type_id
        type_ids = set(self._traci.vehicletype.getIDList())
        if "ego_glosa_type" in type_ids:
            return "ego_glosa_type"
        if "ego_type" in type_ids:
            return "ego_type"
        return "DEFAULT_VEHTYPE"

    def _make_fcd_output_path(self) -> Optional[Path]:
        if not self.config.fcd_output_dir:
            return None
        seed_part = "none" if self._active_seed is None else str(int(self._active_seed))
        return (
            Path(self.config.fcd_output_dir)
            / f"episode_{self._fcd_episode_index:05d}_seed_{seed_part}.fcd.xml"
        )

    def _wait_for_ego(self):
        for _ in range(self.config.max_depart_delay_steps):
            if self.config.ego_id in self._traci.vehicle.getIDList():
                if self.config.speed_mode is not None:
                    self._traci.vehicle.setSpeedMode(
                        self.config.ego_id, int(self.config.speed_mode)
                    )
                return
            self._traci.simulationStep()
        raise RuntimeError(
            f"Ego vehicle '{self.config.ego_id}' did not depart within "
            f"{self.config.max_depart_delay_steps} simulation steps."
        )

    def _apply_acceleration(self, accel: float):
        ego_id = self.config.ego_id
        speed = float(self._traci.vehicle.getSpeed(ego_id))
        allowed_speed = float(self._traci.vehicle.getAllowedSpeed(ego_id))
        speed_cap = min(allowed_speed, self.config.max_speed)
        next_speed = np.clip(speed + accel * self.config.step_length, 0.0, speed_cap)
        self._traci.vehicle.setSpeed(ego_id, float(next_speed))

    def _get_observation(self) -> np.ndarray:
        raw = self._get_raw_observation_array()
        if not self.config.normalize_observation:
            return raw.astype(np.float32)

        scale = np.asarray(
            [
                self.config.max_position,
                self.config.max_speed,
                self.config.max_front_dist,
                self.config.max_speed,
                max(abs(a) for a in self.config.action_accels),
                self.config.max_tl_dist,
                self.config.max_tl_time,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(raw / scale, -1.0, 1.0).astype(np.float32)

    def _get_raw_observation_array(self) -> np.ndarray:
        ego_id = self.config.ego_id
        traci = self._traci

        position = float(traci.vehicle.getDistance(ego_id))
        speed = float(traci.vehicle.getSpeed(ego_id))

        front_dist = self.config.max_front_dist
        delta_v = 0.0
        leader = traci.vehicle.getLeader(ego_id, self.config.max_front_dist)
        if leader is not None:
            leader_id, gap = leader
            front_dist = min(float(gap), self.config.max_front_dist)
            front_speed = float(traci.vehicle.getSpeed(leader_id))
            delta_v = speed - front_speed

        tl_dist, tl_remaining, tl_red, tl_yellow, tl_green = self._next_tls_features()

        return np.asarray(
            [
                min(position, self.config.max_position),
                min(speed, self.config.max_speed),
                front_dist,
                np.clip(delta_v, -self.config.max_speed, self.config.max_speed),
                self._prev_action,
                tl_dist,
                tl_remaining,
                tl_red,
                tl_yellow,
                tl_green,
            ],
            dtype=np.float32,
        )

    def _get_raw_observation_dict(self) -> Dict[str, object]:
        names = [
            "position",
            "speed",
            "front_distance",
            "delta_v_front",
            "previous_action",
            "tls_distance",
            "tls_time_remaining",
            "tls_red",
            "tls_yellow",
            "tls_green",
        ]
        values = self._get_raw_observation_array().tolist()
        raw = dict(zip(names, values))
        position = float(raw["position"])
        tls_distance = float(raw["tls_distance"])
        tls_red = float(raw["tls_red"])
        tls_yellow = float(raw["tls_yellow"])
        tls_green = float(raw["tls_green"])
        raw["tls_position"] = position + tls_distance
        if tls_red:
            raw["tls_state"] = "red"
        elif tls_yellow:
            raw["tls_state"] = "yellow"
        elif tls_green:
            raw["tls_state"] = "green"
        else:
            raw["tls_state"] = "none"
        raw.update(self._neighbor_observation_dict(position))
        return raw

    def _neighbor_observation_dict(self, ego_position: float) -> Dict[str, object]:
        ego_id = self.config.ego_id
        traci = self._traci
        neighbors: Dict[str, object] = {
            "front_vehicle_id": None,
            "front_vehicle_position": None,
            "front_vehicle_speed": None,
            "rear_vehicle_id": None,
            "rear_vehicle_position": None,
            "rear_vehicle_speed": None,
            "rear_distance": self.config.max_front_dist,
            "delta_v_rear": 0.0,
        }

        leader = traci.vehicle.getLeader(ego_id, self.config.max_front_dist)
        if leader is not None:
            leader_id, gap = leader
            neighbors["front_vehicle_id"] = str(leader_id)
            neighbors["front_distance"] = min(float(gap), self.config.max_front_dist)
            neighbors["front_vehicle_position"] = float(
                traci.vehicle.getDistance(leader_id)
            )
            neighbors["front_vehicle_speed"] = float(traci.vehicle.getSpeed(leader_id))

        follower = self._get_follower(ego_id)
        if follower is not None:
            follower_id, gap = follower
            if follower_id:
                rear_speed = float(traci.vehicle.getSpeed(follower_id))
                ego_speed = float(traci.vehicle.getSpeed(ego_id))
                neighbors["rear_vehicle_id"] = str(follower_id)
                neighbors["rear_distance"] = min(float(gap), self.config.max_front_dist)
                neighbors["rear_vehicle_position"] = float(
                    traci.vehicle.getDistance(follower_id)
                )
                neighbors["rear_vehicle_speed"] = rear_speed
                neighbors["delta_v_rear"] = ego_speed - rear_speed
        return neighbors

    def _get_follower(self, ego_id: str):
        try:
            follower = self._traci.vehicle.getFollower(ego_id, self.config.max_front_dist)
        except (AttributeError, TypeError):
            return None
        if follower is None:
            return None
        follower_id, gap = follower
        if follower_id in ("", None) or float(gap) < 0:
            return None
        return follower_id, gap

    def _current_info(self, action_accel: float, arrived: bool) -> Dict[str, object]:
        return {
            "arrived": arrived,
            "episode_step": self._steps,
            "simulation_time": float(self._traci.simulation.getTime()),
            "sumo_seed": self._active_seed,
            "ego_depart_time": self._ego_depart_time,
            "action_accel": float(action_accel),
            **self._energy_info(arrived),
            "raw_observation": self._get_raw_observation_dict() if not arrived else {},
        }

    def _energy_info(self, arrived: bool) -> Dict[str, float]:
        if arrived or self.config.ego_id not in self._traci.vehicle.getIDList():
            return {
                "fuel_consumption_mg_s": 0.0,
                "electricity_consumption_wh_s": 0.0,
                "energy_consumption_kj_s": 0.0,
            }

        fuel_mg_s = self._safe_vehicle_value("getFuelConsumption")
        electricity_wh_s = self._safe_vehicle_value("getElectricityConsumption")
        energy_kj_s = 0.0
        has_energy = False
        if np.isfinite(fuel_mg_s):
            energy_kj_s += (
                fuel_mg_s
                / 1_000_000.0
                * FUEL_LOWER_HEATING_VALUE_KJ_PER_KG
            )
            has_energy = True
        if np.isfinite(electricity_wh_s):
            energy_kj_s += electricity_wh_s * 3.6
            has_energy = True
        return {
            "fuel_consumption_mg_s": float(fuel_mg_s),
            "electricity_consumption_wh_s": float(electricity_wh_s),
            "energy_consumption_kj_s": float(energy_kj_s if has_energy else np.nan),
        }

    def _safe_vehicle_value(self, method_name: str) -> float:
        method = getattr(self._traci.vehicle, method_name, None)
        if method is None:
            return float("nan")
        try:
            return float(method(self.config.ego_id))
        except Exception:
            return float("nan")

    def _next_tls_features(self) -> Tuple[float, float, float, float, float]:
        next_tls: Sequence[Tuple[str, int, float, str]] = self._traci.vehicle.getNextTLS(
            self.config.ego_id
        )
        if not next_tls:
            return self.config.max_tl_dist, self.config.max_tl_time, 0.0, 0.0, 0.0

        tls_id, _tls_index, dist, state = next_tls[0]
        dist = min(float(dist), self.config.max_tl_dist)
        remaining = self._traci.trafficlight.getNextSwitch(tls_id)
        remaining = max(0.0, float(remaining) - float(self._traci.simulation.getTime()))
        remaining = min(remaining, self.config.max_tl_time)

        state = str(state).lower()
        is_red = float("r" in state)
        is_yellow = float("y" in state)
        is_green = float("g" in state)
        return dist, remaining, is_red, is_yellow, is_green
