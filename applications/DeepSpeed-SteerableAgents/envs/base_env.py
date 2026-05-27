"""Minimal environment interface used by the rollout collector."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseEnv(ABC):
    """A minimal env API. Not gym-compatible on purpose (smaller surface)."""

    #: Number of discrete actions available to the agent.
    num_actions: int = 0
    #: Dimensionality of the observation vector.
    obs_dim: int = 0

    @abstractmethod
    def reset(self) -> List[float]:
        """Reset the env and return the initial observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        """Take one action.

        Returns ``(next_obs, reward, done, info)``.
        ``info`` may contain ``"oracle_action"`` and ``"is_trap"`` to help
        the simulated teacher.
        """

    @abstractmethod
    def render(self) -> str:
        """Return a short string representation (debug only)."""
