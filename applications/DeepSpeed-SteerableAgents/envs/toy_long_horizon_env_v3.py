"""Harder long-horizon toy env (V3) with multiple critical decision nodes.

Design goals:
- A single intervention can only fix one critical node.
- Success depends on multiple independently challenging decisions.
- Difficulty is tunable via knobs (critical-node count, stochasticity, etc.).

Observation layout (length ``obs_dim``)::

    [ step/episode_steps,
      critical_passes/required_critical_passes,
      one_hot(last_action; size=num_actions),
      one_hot(current_hint_action; size=num_actions),
      is_critical_step,
      remaining_critical_nodes/num_critical_nodes ]

At each critical step there is a hidden oracle action and a hidden trap action.
The observation exposes a *noisy hint* action. The teacher can read the exact
oracle/trap only through ``get_probe_info()``.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv


class ToyLongHorizonEnvV3(BaseEnv):
    """Harder multi-critical-node long-horizon environment."""

    def __init__(
        self,
        num_actions: int = 4,
        horizon: int = 40,
        seed: Optional[int] = None,
        episode_steps: Optional[int] = None,
        num_critical_nodes: int = 4,
        required_critical_passes: Optional[int] = None,
        stochasticity: float = 0.25,
        transition_noise: float = 0.10,
    ) -> None:
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.episode_steps = int(episode_steps if episode_steps is not None else self.horizon)
        self.num_critical_nodes = max(1, int(num_critical_nodes))
        self.required_critical_passes = int(
            required_critical_passes
            if required_critical_passes is not None
            else self.num_critical_nodes
        )
        self.stochasticity = float(max(0.0, min(1.0, stochasticity)))
        self.transition_noise = float(max(0.0, min(1.0, transition_noise)))

        # obs = step + progress + one_hot(last) + one_hot(hint) + 2 scalar flags
        self.obs_dim = 2 + self.num_actions + self.num_actions + 2

        self._rng = random.Random(seed)
        self._done = True
        self._step = 0
        self._last_action = -1
        self._critical_passes = 0
        self._critical_seen = 0

        self._critical_steps: List[int] = []
        self._good_actions: List[int] = []
        self._trap_actions: List[int] = []

    # ------------------------------------------------------------------ utils
    def _build_episode_spec(self) -> None:
        # Evenly spread critical nodes across the episode timeline.
        slots = []
        for i in range(self.num_critical_nodes):
            pos = int(round((i + 1) * self.episode_steps / (self.num_critical_nodes + 1)))
            pos = max(0, min(self.episode_steps - 1, pos))
            slots.append(pos)
        self._critical_steps = sorted(set(slots))

        # If collisions reduced the number of unique slots, pad with random slots.
        while len(self._critical_steps) < self.num_critical_nodes:
            c = self._rng.randrange(self.episode_steps)
            if c not in self._critical_steps:
                self._critical_steps.append(c)
        self._critical_steps.sort()

        self._good_actions = []
        self._trap_actions = []
        for _ in range(self.num_critical_nodes):
            good = self._rng.randrange(self.num_actions)
            choices = [a for a in range(self.num_actions) if a != good]
            trap = self._rng.choice(choices)
            self._good_actions.append(good)
            self._trap_actions.append(trap)

    def _critical_index_at_step(self, step: int) -> Optional[int]:
        try:
            return self._critical_steps.index(step)
        except ValueError:
            return None

    def _current_hint_action(self) -> int:
        idx = self._critical_index_at_step(self._step)
        if idx is None:
            return -1
        good = self._good_actions[idx]
        if self._rng.random() >= self.stochasticity:
            return good
        # Noisy hint: can be wrong.
        choices = [a for a in range(self.num_actions) if a != good]
        return self._rng.choice(choices)

    def _obs(self) -> List[float]:
        hint = self._current_hint_action()
        obs = [
            self._step / max(1, self.episode_steps),
            self._critical_passes / max(1, self.required_critical_passes),
        ]

        last_oh = [0.0] * self.num_actions
        if 0 <= self._last_action < self.num_actions:
            last_oh[self._last_action] = 1.0
        obs.extend(last_oh)

        hint_oh = [0.0] * self.num_actions
        if 0 <= hint < self.num_actions:
            hint_oh[hint] = 1.0
        obs.extend(hint_oh)

        idx = self._critical_index_at_step(self._step)
        obs.append(1.0 if idx is not None else 0.0)
        remaining = max(0, self.num_critical_nodes - self._critical_seen)
        obs.append(remaining / max(1, self.num_critical_nodes))
        return obs

    # ------------------------------------------------------------------ API
    def reset(self) -> List[float]:
        self._done = False
        self._step = 0
        self._last_action = -1
        self._critical_passes = 0
        self._critical_seen = 0
        self._build_episode_spec()
        return self._obs()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a finished episode; call reset().")
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action {action} out of range")

        reward = 0.0
        idx = self._critical_index_at_step(self._step)
        is_critical = idx is not None
        oracle_action = None
        trap_action = None
        is_trap = False

        effective_action = int(action)
        if is_critical and self.transition_noise > 0 and self._rng.random() < self.transition_noise:
            effective_action = self._rng.randrange(self.num_actions)

        if is_critical:
            oracle_action = int(self._good_actions[idx])
            trap_action = int(self._trap_actions[idx])
            self._critical_seen += 1
            if effective_action == oracle_action:
                self._critical_passes += 1
                reward += 1.0
            else:
                reward -= 0.1
                if effective_action == trap_action:
                    reward -= 0.4
                    is_trap = True

        self._step += 1
        self._last_action = int(action)

        success = False
        if self._step >= self.episode_steps:
            self._done = True
            success = self._critical_passes >= self.required_critical_passes
            if success:
                reward += 5.0

        info = {
            "oracle_action": oracle_action,
            "trap_action": trap_action,
            "is_trap": is_trap,
            "is_critical_node": bool(is_critical),
            "critical_node_index": int(idx) if idx is not None else None,
            "critical_passes": int(self._critical_passes),
            "critical_seen": int(self._critical_seen),
            "num_critical_nodes": int(self.num_critical_nodes),
            "success": success,
        }
        return self._obs(), reward, self._done, info

    def get_probe_info(self) -> Dict[str, Any]:
        idx = self._critical_index_at_step(self._step)
        is_critical = idx is not None
        oracle = int(self._good_actions[idx]) if is_critical else None
        trap = int(self._trap_actions[idx]) if is_critical else None
        return {
            "oracle_action": oracle,
            "trap_action": trap,
            "is_critical_node": bool(is_critical),
            "critical_node_index": int(idx) if idx is not None else None,
            "num_critical_nodes": int(self.num_critical_nodes),
            "critical_passes": int(self._critical_passes),
            "critical_seen": int(self._critical_seen),
            "remaining_critical_nodes": int(max(0, self.num_critical_nodes - self._critical_seen)),
        }

    def render(self) -> str:
        return (
            f"V3 step={self._step}/{self.episode_steps} "
            f"passes={self._critical_passes}/{self.required_critical_passes} "
            f"seen={self._critical_seen}/{self.num_critical_nodes} "
            f"critical_steps={self._critical_steps} done={self._done}"
        )
