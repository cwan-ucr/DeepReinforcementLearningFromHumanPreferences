from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pickle
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

    def __init__(self, obs_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        layers: list[nn.Module] = [
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

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


class RewardEnsemble(nn.Module):
    """Mean prediction from independently initialized step reward models."""

    def __init__(
        self,
        obs_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.members = nn.ModuleList(
            [
                StepRewardModel(obs_dim, hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, obs: torch.Tensor, action_value: torch.Tensor) -> torch.Tensor:
        rewards = torch.stack(
            [member(obs, action_value) for member in self.members],
            dim=0,
        )
        return rewards.mean(dim=0)

    def segment_return(self, segment: Sequence[Tuple[Sequence[float], float]]) -> torch.Tensor:
        returns = torch.stack([member.segment_return(segment) for member in self.members])
        return returns.mean()

    @torch.no_grad()
    def predict_step_reward(self, obs: Sequence[float], action_value: float) -> float:
        self.eval()
        obs_t = torch.tensor([obs], dtype=torch.float32)
        action_t = torch.tensor([action_value], dtype=torch.float32)
        return float(self.forward(obs_t, action_t).item())


RewardModelType = Union[StepRewardModel, RewardEnsemble]


def preference_loss(model: RewardModelType, examples: Iterable[PreferenceExample]) -> torch.Tensor:
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
    model: RewardModelType,
    examples: Sequence[PreferenceExample],
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
) -> List[float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
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


def train_reward_ensemble(
    ensemble: RewardEnsemble,
    examples: Sequence[PreferenceExample],
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    bootstrap: bool = True,
) -> List[float]:
    """Train each ensemble member independently, optionally on bootstrap samples."""

    member_losses: List[List[float]] = []
    if not examples:
        return [0.0]
    for member in ensemble.members:
        if bootstrap:
            sample_indices = np.random.randint(0, len(examples), size=len(examples))
            member_examples = [examples[int(i)] for i in sample_indices]
        else:
            member_examples = list(examples)
        member_losses.append(
            train_reward_model(
                member,
                member_examples,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        )

    max_len = max(len(losses) for losses in member_losses)
    losses = []
    for epoch in range(max_len):
        values = [
            member_epoch_losses[epoch]
            for member_epoch_losses in member_losses
            if epoch < len(member_epoch_losses)
        ]
        losses.append(float(np.mean(values)) if values else 0.0)
    return losses


def save_reward_checkpoint(
    model: RewardModelType,
    path: str | Path,
    *,
    metadata: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, RewardEnsemble):
        model_type = "ensemble"
        config = {
            "obs_dim": model.obs_dim,
            "ensemble_size": model.ensemble_size,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
        }
    else:
        model_type = "single"
        config = {
            "obs_dim": model.obs_dim,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
        }
    torch.save(
        {
            "format": "sumo_rlhf_reward_checkpoint_v2",
            "model_type": model_type,
            "config": config,
            "metadata": metadata or {},
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_reward_checkpoint(
    path: str | Path,
    obs_dim: int | None = None,
    *,
    hidden_dim: int = 128,
) -> RewardModelType:
    checkpoint = load_torch_checkpoint(path)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "sumo_rlhf_reward_checkpoint_v2":
        config = dict(checkpoint.get("config", {}))
        model_type = checkpoint.get("model_type", "single")
        if model_type == "ensemble":
            model: RewardModelType = RewardEnsemble(
                obs_dim=int(config["obs_dim"]),
                ensemble_size=int(config.get("ensemble_size", 5)),
                hidden_dim=int(config.get("hidden_dim", hidden_dim)),
                dropout=float(config.get("dropout", 0.0)),
            )
        else:
            model = StepRewardModel(
                obs_dim=int(config["obs_dim"]),
                hidden_dim=int(config.get("hidden_dim", hidden_dim)),
                dropout=float(config.get("dropout", 0.0)),
            )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    if obs_dim is None:
        raise ValueError(
            "obs_dim is required when loading legacy reward checkpoints saved as a raw state_dict."
        )
    model = StepRewardModel(obs_dim=obs_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_torch_checkpoint(path: str | Path):
    """Load local project checkpoints across PyTorch weights_only defaults."""

    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)
