"""Toy policy heads.

A small MLP student that maps observation -> action logits. Kept tiny so it
trains in seconds on CPU. Real experiments would swap this for an LLM or a
larger Transformer policy.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Simple feed-forward policy head with categorical action distribution."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.num_actions = num_actions
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape ``(B, num_actions)``."""
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False) -> Tuple[int, torch.Tensor, float]:
        """Return ``(action_id, logits, entropy)`` for a single obs vector."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if greedy:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            action = int(torch.multinomial(probs, num_samples=1).item())
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).item())
        return action, logits.squeeze(0), entropy


class OracleTeacher:
    """Trivial 'teacher' that knows the oracle action from env info.

    Useful only as a stand-in for a real strong-model or human teacher.
    """

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions

    def teacher_logits(self, oracle_action: int) -> torch.Tensor:
        logits = torch.full((self.num_actions,), -2.0)
        logits[oracle_action] = 2.0
        return logits
