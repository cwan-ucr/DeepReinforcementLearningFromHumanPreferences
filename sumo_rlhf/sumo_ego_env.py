from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError("sumo_rlhf currently expects gym to be installed.") from exc


@dataclass
class SumoEgoConfig:
    """Configuration for a single-lane arterial ego-vehicle SUMO environment."""

    sumo_cfg: str
    ego_id: str = "ego"
    gui: bool = False
    step_length: float = 1.0
    max_episode_steps: int = 300
    max_depart_delay_steps: int = 1000
    action_accels: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.0, 2.0)
    max_position: float = 3000.0
    max_speed: float = 25.0
    max_front_dist: float = 120.0
    max_tl_dist: float = 500.0
    max_tl_time: float = 120.0
    normalize_observation: bool = True
    speed_mode: Optional[int] = None


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
        self._prev_action = 0.0
        self._last_obs = np.zeros(10, dtype=np.float32)

        self.action_space = spaces.Discrete(len(config.action_accels))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

    def reset(self):
        self.close()
        self._start_sumo()
        self._steps = 0
        self._prev_action = 0.0
        self._wait_for_ego()
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
            "arrived": arrived,
            "timeout": timeout,
            "action_accel": accel,
            "raw_observation": self._get_raw_observation_dict() if not arrived else {},
        }
        return obs, 0.0, done, info

    def close(self):
        if self._traci is not None:
            try:
                self._traci.close(False)
            finally:
                self._traci = None

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)

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
        traci.start(cmd)
        self._traci = traci
        self._sumolib = sumolib

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

    def _get_raw_observation_dict(self) -> Dict[str, float]:
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
        return dict(zip(names, self._get_raw_observation_array().tolist()))

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
        is_green = float(("g" in state) or ("G" in state))
        return dist, remaining, is_red, is_yellow, is_green

