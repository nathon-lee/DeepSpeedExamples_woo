"""Learnable variant of the toy long-horizon env.

Compared to ``ToyLongHorizonEnv``, V2:

- Fixes ``good_action`` and ``trap_action`` *per episode* (sampled in reset)
  instead of resampling every step. This gives the agent a stable task to
  learn within an episode.
- Exposes both of them as one-hot fields inside the observation, so the
  student policy actually has enough information to recover the optimal
  policy from supervision.
- Keeps progress monotonic (wrong but non-trap actions just waste a step).
- Keeps the same ``step``/``reset``/``info`` contract as the V1 env, so the
  rollout collector, teacher policy and budget controller work unchanged.

Use this env when you want to see distillation loss actually decrease and to
measure a real success-rate uplift from steering.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv


class ToyLongHorizonEnvV2(BaseEnv):
    """Learnable toy long-horizon env (per-episode fixed oracle/trap).

    Observation layout (length ``obs_dim``)::

        [ step/H,
          progress/H,
          one_hot(last_action; size=num_actions),
          one_hot(good_action; size=num_actions),   # task context
          one_hot(trap_action; size=num_actions) ]  # task context
    """

    def __init__(
        self,
        num_actions: int = 4,
        horizon: int = 32,
        seed: Optional[int] = None,
        reveal_good: bool = True,
        reveal_trap: bool = True,
        episode_steps: Optional[int] = None,
        progress_goal: Optional[int] = None,
    ) -> None:
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.reveal_good = bool(reveal_good)
        self.reveal_trap = bool(reveal_trap)
        # `horizon` is the *progress* horizon (target). `episode_steps` is the
        # actual max number of steps the agent is allowed to take. Defaults
        # make them equal (strict all-or-nothing). Setting episode_steps >
        # horizon turns the task into a "budget-of-time" version, where some
        # wasted steps are tolerated -- this is what produces smooth
        # budget-vs-success curves for steering experiments.
        self.progress_goal = int(progress_goal if progress_goal is not None else self.horizon)
        self.episode_steps = int(episode_steps if episode_steps is not None else self.horizon)
        # obs = step + progress + one_hot(last) + optional one_hot(good)/one_hot(trap)
        self.obs_dim = (
            2
            + self.num_actions
            + (self.num_actions if self.reveal_good else 0)
            + (self.num_actions if self.reveal_trap else 0)
        )
        self._rng = random.Random(seed)
        self._step = 0
        self._progress = 0
        self._last_action = -1
        self._good_action = 0
        self._trap_action = 1
        self._done = True

    # ------------------------------------------------------------------ utils
    def _sample_hidden(self) -> None:
        self._good_action = self._rng.randrange(self.num_actions)
        choices = [a for a in range(self.num_actions) if a != self._good_action]
        self._trap_action = self._rng.choice(choices)

    def _obs(self) -> List[float]:
        obs = [
            self._step / max(1, self.episode_steps),
            self._progress / max(1, self.progress_goal),
        ]
        last_oh = [0.0] * self.num_actions
        if 0 <= self._last_action < self.num_actions:
            last_oh[self._last_action] = 1.0
        obs.extend(last_oh)
        if self.reveal_good:
            good_oh = [0.0] * self.num_actions
            good_oh[self._good_action] = 1.0
            obs.extend(good_oh)
        if self.reveal_trap:
            trap_oh = [0.0] * self.num_actions
            trap_oh[self._trap_action] = 1.0
            obs.extend(trap_oh)
        return obs

    # ------------------------------------------------------------------ API
    def reset(self) -> List[float]:
        self._step = 0
        self._progress = 0
        self._last_action = -1
        self._done = False
        self._sample_hidden()  # fixed for the whole episode
        return self._obs()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a finished episode; call reset().")
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action {action} out of range")

        reward = 0.0
        is_trap = action == self._trap_action
        if action == self._good_action:
            self._progress += 1
            reward = 1.0
        elif is_trap:
            self._done = True
            reward = -1.0
        # other actions: waste a step, no reward, no progress.

        self._step += 1
        self._last_action = action

        success = self._progress >= self.progress_goal
        if success:
            self._done = True
            reward += 5.0
        elif self._step >= self.episode_steps:
            self._done = True  # ran out of time

        info = {
            "oracle_action": self._good_action,
            "trap_action": self._trap_action,
            "is_trap": is_trap,
            "progress": self._progress,
            "success": success,
        }
        return self._obs(), reward, self._done, info

    def get_probe_info(self) -> Dict[str, Any]:
        return {
            "oracle_action": int(self._good_action),
            "trap_action": int(self._trap_action),
            "progress": int(self._progress),
        }

    def render(self) -> str:
        return (
            f"V2 step={self._step}/{self.horizon} progress={self._progress} "
            f"good={self._good_action} trap={self._trap_action} "
            f"last={self._last_action} done={self._done}"
        )
