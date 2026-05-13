from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


Preference = int  # 0 means left preferred, 1 means right preferred, 2 means neutral.


@dataclass
class PreferenceExample:
    left: List[Tuple[List[float], float]]
    right: List[Tuple[List[float], float]]
    preference: Preference


class StepRewardModel(nn.Module):
    """Predicts scalar reward r_theta(obs, action) for each trajectory step."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action_value: torch.Tensor) -> torch.Tensor:
        if action_value.ndim == 1:
            action_value = action_value.unsqueeze(-1)
        x = torch.cat([obs, action_value], dim=-1)
        return self.net(x).squeeze(-1)

    def segment_return(self, segment: Sequence[Tuple[Sequence[float], float]]) -> torch.Tensor:
        obs = torch.tensor([step[0] for step in segment], dtype=torch.float32)
        actions = torch.tensor([step[1] for step in segment], dtype=torch.float32)
        return self.forward(obs, actions).sum()

    @torch.no_grad()
    def predict_step_reward(self, obs: Sequence[float], action_value: float) -> float:
        self.eval()
        obs_t = torch.tensor([obs], dtype=torch.float32)
        action_t = torch.tensor([action_value], dtype=torch.float32)
        return float(self.forward(obs_t, action_t).item())


def preference_loss(model: StepRewardModel, examples: Iterable[PreferenceExample]) -> torch.Tensor:
    losses = []
    for example in examples:
        left_return = model.segment_return(example.left)
        right_return = model.segment_return(example.right)
        logits = torch.stack([left_return, right_return], dim=0)

        if example.preference == 0:
            target = torch.tensor([1.0, 0.0], dtype=torch.float32)
        elif example.preference == 1:
            target = torch.tensor([0.0, 1.0], dtype=torch.float32)
        else:
            target = torch.tensor([0.5, 0.5], dtype=torch.float32)

        losses.append(-(target * F.log_softmax(logits, dim=0)).sum())

    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def train_reward_model(
    model: StepRewardModel,
    examples: Sequence[PreferenceExample],
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: List[float] = []
    indices = np.arange(len(examples))

    for _ in range(epochs):
        np.random.shuffle(indices)
        epoch_losses = []
        for start in range(0, len(indices), batch_size):
            batch = [examples[int(i)] for i in indices[start : start + batch_size]]
            loss = preference_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    return losses

