from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pickle
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 64


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_count: int, hidden_dim: int = 128):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_count)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_features = self.actor_body(obs)
        critic_features = self.critic_body(obs)
        logits = self.actor_head(actor_features)
        values = self.critic_head(critic_features).squeeze(-1)
        return logits, values


class PPORolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ):
        self.obs.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        if action_mask is None:
            raise ValueError("action_mask must be provided for PPO rollouts.")
        self.action_masks.append(np.asarray(action_mask, dtype=np.float32).copy())

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_count: int,
        config: PPOConfig | None = None,
    ):
        self.config = config or PPOConfig()
        self.action_count = action_count
        self.model = ActorCritic(obs_dim, action_count)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.rollout = PPORolloutBuffer()
        self.last_update_loss: float | None = None
        self.last_policy_loss: float | None = None
        self.last_value_loss: float | None = None
        self.last_entropy: float | None = None

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, action_mask: np.ndarray | None = None
    ) -> tuple[int, float, float]:
        self.model.eval()
        obs_t = torch.tensor(np.asarray([obs]), dtype=torch.float32)
        logits, value = self.model(obs_t)
        if action_mask is not None:
            mask_t = torch.tensor(np.asarray([action_mask]), dtype=torch.bool)
            logits = logits.masked_fill(~mask_t, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ):
        if action_mask is None:
            action_mask = np.ones(self.action_count, dtype=np.float32)
        self.rollout.add(obs, action, log_prob, value, reward, done, action_mask)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "sumo_rlhf_ppo_policy_v2",
                "config": {
                    "gamma": float(self.config.gamma),
                    "gae_lambda": float(self.config.gae_lambda),
                    "learning_rate": float(self.config.learning_rate),
                    "clip_ratio": float(self.config.clip_ratio),
                    "value_coef": float(self.config.value_coef),
                    "entropy_coef": float(self.config.entropy_coef),
                    "max_grad_norm": float(self.config.max_grad_norm),
                    "update_epochs": int(self.config.update_epochs),
                    "batch_size": int(self.config.batch_size),
                },
                "action_count": int(self.action_count),
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path):
        checkpoint = load_torch_checkpoint(path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = upgrade_legacy_shared_backbone_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def update(self, next_value: float = 0.0) -> dict[str, float] | None:
        if len(self.rollout) == 0:
            return None

        obs_t = torch.tensor(np.asarray(self.rollout.obs), dtype=torch.float32)
        actions_t = torch.tensor(self.rollout.actions, dtype=torch.int64)
        old_log_probs_t = torch.tensor(self.rollout.log_probs, dtype=torch.float32)
        action_masks_t = torch.tensor(
            np.asarray(self.rollout.action_masks), dtype=torch.bool
        )
        old_values = np.asarray(self.rollout.values + [float(next_value)], dtype=np.float32)
        rewards = np.asarray(self.rollout.rewards, dtype=np.float32)
        dones = np.asarray(self.rollout.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + self.config.gamma * old_values[step + 1] * nonterminal
                - old_values[step]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[step] = gae
        returns = advantages + old_values[:-1]

        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )
        returns_t = torch.tensor(returns, dtype=torch.float32)

        indices = np.arange(len(rewards))
        loss_values = []
        policy_losses = []
        value_losses = []
        entropies = []

        self.model.train()
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                batch_idx_t = torch.tensor(batch_idx, dtype=torch.int64)

                logits, values = self.model(obs_t[batch_idx_t])
                logits = logits.masked_fill(~action_masks_t[batch_idx_t], -1e9)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions_t[batch_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx_t])
                unclipped = ratio * advantages_t[batch_idx_t]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * advantages_t[batch_idx_t]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns_t[batch_idx_t])
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                loss_values.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.rollout.clear()
        metrics = {
            "loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
        self.last_update_loss = metrics["loss"]
        self.last_policy_loss = metrics["policy_loss"]
        self.last_value_loss = metrics["value_loss"]
        self.last_entropy = metrics["entropy"]
        return metrics


def load_torch_checkpoint(path: str | Path):
    """Load local project checkpoints across PyTorch weights_only defaults."""

    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def upgrade_legacy_shared_backbone_state_dict(state_dict: dict) -> dict:
    """Map legacy shared-backbone PPO checkpoints to split actor/critic backbones."""
    keys = set(state_dict.keys())
    if "actor_body.0.weight" in keys or "critic_body.0.weight" in keys:
        return state_dict
    if "shared.0.weight" not in keys:
        return state_dict

    migrated = dict(state_dict)
    legacy_to_new = {
        "shared.0.weight": ("actor_body.0.weight", "critic_body.0.weight"),
        "shared.0.bias": ("actor_body.0.bias", "critic_body.0.bias"),
        "shared.2.weight": ("actor_body.2.weight", "critic_body.2.weight"),
        "shared.2.bias": ("actor_body.2.bias", "critic_body.2.bias"),
        "policy_head.weight": ("actor_head.weight",),
        "policy_head.bias": ("actor_head.bias",),
        "value_head.weight": ("critic_head.weight",),
        "value_head.bias": ("critic_head.bias",),
    }
    for old_key, new_keys in legacy_to_new.items():
        if old_key in state_dict:
            for new_key in new_keys:
                migrated[new_key] = state_dict[old_key]
    return migrated
