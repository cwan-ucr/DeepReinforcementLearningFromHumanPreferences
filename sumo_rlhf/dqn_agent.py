from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 1e-3
    memory_size: int = 100_000
    batch_size: int = 64
    exploration_max: float = 1.0
    exploration_min: float = 0.05
    exploration_decay: float = 0.995


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_count: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, action_count: int, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        self.action_count = action_count
        self.exploration_rate = self.config.exploration_max
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.config.memory_size
        )
        self.model = QNetwork(obs_dim, action_count)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()

    def act(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_count)
        with torch.no_grad():
            obs_t = torch.tensor([obs], dtype=torch.float32)
            return int(torch.argmax(self.model(obs_t), dim=1).item())

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.memory.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_obs_t = torch.tensor(np.asarray(next_obs), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(obs_t).gather(1, actions_t).squeeze(-1)
        with torch.no_grad():
            next_q = self.model(next_obs_t).max(dim=1).values
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.config.exploration_decay
        self.exploration_rate = max(
            self.config.exploration_min, self.exploration_rate
        )
        return float(loss.item())

