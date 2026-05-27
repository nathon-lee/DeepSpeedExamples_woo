"""Toy long-horizon environment.

Semantics
---------
The agent must execute a long-horizon plan of ``horizon`` steps. At each
step there is a single hidden "good" action that advances progress by one
unit; one "trap" action that ends the episode in failure; the remaining
actions are no-ops (cost a step but do not advance or fail).

This is intentionally minimal but exhibits the structural problems we care
about for steering research:

- Long horizons (default 32 steps): credit assignment matters.
- Catastrophic mistakes (traps): vetoes are valuable.
- Progress is partially observable: progress-update interventions help.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv


class ToyLongHorizonEnv(BaseEnv):
    """A discrete, multi-step toy env with traps and progress.

    Observation (length ``obs_dim``):
      [norm_step, norm_progress, *one_hot(last_action)]
    Hidden state:
      - ``good_action``: the action that advances progress this step.
      - ``trap_action``: the action that fails the episode this step.
    Both are resampled every step using a seeded RNG so that an oracle teacher
    has non-trivial information.
    """

    def __init__(
        self,
        num_actions: int = 4,
        horizon: int = 32,
        seed: Optional[int] = None,
    ) -> None:
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.obs_dim = 2 + self.num_actions
        self._rng = random.Random(seed)
        self._step = 0
        self._progress = 0
        self._last_action = -1
        self._good_action = 0
        self._trap_action = 1
        self._done = True

    # ------------------------------------------------------------------ utils
    def _resample_hidden(self) -> None:
        self._good_action = self._rng.randrange(self.num_actions)
        # ensure trap != good
        choices = [a for a in range(self.num_actions) if a != self._good_action]
        self._trap_action = self._rng.choice(choices)

    def _obs(self) -> List[float]:
        obs = [
            self._step / max(1, self.horizon),
            self._progress / max(1, self.horizon),
        ]
        one_hot = [0.0] * self.num_actions
        if 0 <= self._last_action < self.num_actions:
            one_hot[self._last_action] = 1.0
        obs.extend(one_hot)
        return obs

    # ------------------------------------------------------------------ API
    def reset(self) -> List[float]:
        self._step = 0
        self._progress = 0
        self._last_action = -1
        self._done = False
        self._resample_hidden()
        return self._obs()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a finished episode; call reset().")
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action {action} out of range")

        oracle_action = self._good_action
        is_trap = action == self._trap_action

        reward = 0.0
        if action == self._good_action:
            self._progress += 1
            reward = 1.0
        elif is_trap:
            self._done = True
            reward = -1.0

        self._step += 1
        self._last_action = action

        success = self._progress >= self.horizon
        if success:
            self._done = True
            reward += 5.0
        elif self._step >= self.horizon:
            self._done = True  # ran out of time

        info = {
            "oracle_action": oracle_action,
            "trap_action": self._trap_action,
            "is_trap": is_trap,
            "progress": self._progress,
            "success": success,
        }

        if not self._done:
            self._resample_hidden()
        return self._obs(), reward, self._done, info

    def render(self) -> str:
        return (
            f"step={self._step}/{self.horizon} progress={self._progress} "
            f"good={self._good_action} trap={self._trap_action} "
            f"last={self._last_action} done={self._done}"
        )
