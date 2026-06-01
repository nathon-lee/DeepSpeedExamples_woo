"""Harder long-horizon toy env (V3) with multiple critical decision nodes.

Design goals:
- Multiple independently challenging critical decisions.
- Success is *soft*: missing a critical node penalizes but does not always
  doom the trajectory (see ``failure_softness`` / ``required_critical_passes``).
- A single intervention fixes the current critical node locally, and (via
  ``intervention_effect_span``) may grant a short local follow-up advantage,
  but never globally solves the task.
- Difficulty is tunable via knobs and named presets (easy / medium / hard).

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

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv


# Named difficulty presets. Explicit kwargs always override preset values.
#
# Calibration note (2026-06): medium previously used 4 critical nodes, so B=4
# and B=8 both spent at most four useful interventions and saturated at the same
# student success (~49%).  The paper-facing medium preset now exposes six local
# critical decisions and requires four passes, giving B=4/B=8 room to separate
# while keeping B=0 near zero.
DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": dict(
        num_critical_nodes=3,
        horizon=18,
        stochasticity=0.12,
        transition_noise=0.06,
        intervention_effect_span=2,
        failure_softness="high",
    ),
    "medium": dict(
        num_critical_nodes=6,
        horizon=32,
        stochasticity=0.18,
        transition_noise=0.08,
        intervention_effect_span=1,
        failure_softness="medium",
    ),
    "hard": dict(
        num_critical_nodes=7,
        horizon=36,
        stochasticity=0.25,
        transition_noise=0.12,
        intervention_effect_span=1,
        failure_softness="low",
    ),
}

# failure_softness -> fraction of critical nodes that must be passed to succeed.
# Values are intentionally fractional so they scale across presets:
#   high   + 3 nodes -> 2/3 (easy, not trivial)
#   medium + 6 nodes -> 4/6 (paper-facing target)
#   low    + 7 nodes -> 6/7 (challenging, but still soft-failure)
_SOFTNESS_REQUIRED_FRACTION = {
    "high": 0.67,
    "medium": 0.67,
    "low": 0.80,
}


def resolve_difficulty(difficulty: Optional[str]) -> Dict[str, Any]:
    """Return the preset kwargs for a named difficulty, or {} when unset."""
    if not difficulty:
        return {}
    key = str(difficulty).lower()
    if key not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; "
            f"expected one of {sorted(DIFFICULTY_PRESETS)}"
        )
    return dict(DIFFICULTY_PRESETS[key])


class ToyLongHorizonEnvV3(BaseEnv):
    """Mid-difficulty multi-critical-node long-horizon environment."""

    def __init__(
        self,
        num_actions: int = 4,
        horizon: int = 32,
        seed: Optional[int] = None,
        episode_steps: Optional[int] = None,
        num_critical_nodes: Optional[int] = None,
        required_critical_passes: Optional[int] = None,
        stochasticity: Optional[float] = None,
        transition_noise: Optional[float] = None,
        difficulty: Optional[str] = None,
        intervention_effect_span: Optional[int] = None,
        failure_softness: Optional[str] = None,
    ) -> None:
        preset = resolve_difficulty(difficulty)

        def pick(name: str, explicit, default):
            if explicit is not None:
                return explicit
            if name in preset:
                return preset[name]
            return default

        self.difficulty = (difficulty or "").lower()
        self.num_actions = int(num_actions)
        # ``horizon`` has a non-None default, so only honour the preset when the
        # caller left it at a documented V3 default and selected a difficulty.
        if difficulty and horizon in {24, 28, 32}:
            self.horizon = int(preset.get("horizon", horizon))
        else:
            self.horizon = int(horizon)
        self.num_critical_nodes = max(1, int(pick("num_critical_nodes", num_critical_nodes, 6)))
        self.stochasticity = float(max(0.0, min(1.0, pick("stochasticity", stochasticity, 0.18))))
        self.transition_noise = float(max(0.0, min(1.0, pick("transition_noise", transition_noise, 0.08))))
        self.intervention_effect_span = max(1, int(pick("intervention_effect_span", intervention_effect_span, 1)))
        self.failure_softness = str(pick("failure_softness", failure_softness, "medium")).lower()

        self.episode_steps = int(episode_steps if episode_steps is not None else self.horizon)

        # Soft-failure: required passes default derives from failure_softness.
        if required_critical_passes is not None:
            self.required_critical_passes = int(required_critical_passes)
        else:
            frac = _SOFTNESS_REQUIRED_FRACTION.get(self.failure_softness, 0.67)
            self.required_critical_passes = max(
                1, int(math.ceil(frac * self.num_critical_nodes))
            )
        self.required_critical_passes = min(
            self.required_critical_passes, self.num_critical_nodes
        )

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

        # Local intervention-effect bookkeeping.
        self._intervention_pending = False
        self._boost_remaining = 0

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
        # A local intervention boost makes the upcoming hint noise-free.
        if self._boost_remaining > 0:
            return good
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
        self._intervention_pending = False
        self._boost_remaining = 0
        self._build_episode_spec()
        return self._obs()

    def notify_intervention(self, step: Optional[int] = None) -> None:
        """Mark that the *next* ``step`` was steered by a teacher intervention.

        The collector calls this right before executing the (overridden)
        action. It (a) suppresses transition noise so the steered action lands
        reliably, and (b) seeds a local follow-up advantage of
        ``intervention_effect_span - 1`` subsequent critical nodes.
        """
        self._intervention_pending = True

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a finished episode; call reset().")
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action {action} out of range")

        intervened = self._intervention_pending
        self._intervention_pending = False

        reward = 0.0
        idx = self._critical_index_at_step(self._step)
        is_critical = idx is not None
        oracle_action = None
        trap_action = None
        is_trap = False

        effective_action = int(action)
        # Transition noise is suppressed on steered steps and during a boost.
        noise_active = (
            is_critical
            and self.transition_noise > 0
            and not intervened
            and self._boost_remaining <= 0
        )
        if noise_active and self._rng.random() < self.transition_noise:
            effective_action = self._rng.randrange(self.num_actions)

        if is_critical:
            oracle_action = int(self._good_actions[idx])
            trap_action = int(self._trap_actions[idx])
            self._critical_seen += 1

            if intervened:
                # A local intervention seeds a short follow-up advantage.
                self._boost_remaining = max(
                    self._boost_remaining, self.intervention_effect_span - 1
                )
            elif self._boost_remaining > 0:
                # Consume one unit of the local follow-up advantage.
                self._boost_remaining -= 1

            if effective_action == oracle_action:
                self._critical_passes += 1
                reward += 1.0
            else:
                # Soft penalty: missing a node hurts but does not end the run.
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
            "required_critical_passes": int(self.required_critical_passes),
            "success": success,
        }
        return self._obs(), reward, self._done, info

    def get_probe_info(self) -> Dict[str, Any]:
        idx = self._critical_index_at_step(self._step)
        is_critical = idx is not None
        oracle = int(self._good_actions[idx]) if is_critical else None
        trap = int(self._trap_actions[idx]) if is_critical else None
        remaining_required = max(
            0, self.required_critical_passes - self._critical_passes
        )
        remaining_nodes = max(0, self.num_critical_nodes - self._critical_seen)
        # Risk is high when we still *need* passes and few nodes remain.
        risk = 0.0
        if remaining_nodes > 0:
            risk = min(1.0, remaining_required / max(1, remaining_nodes))
        return {
            "oracle_action": oracle,
            "trap_action": trap,
            "is_critical_node": bool(is_critical),
            "critical_node_index": int(idx) if idx is not None else None,
            "num_critical_nodes": int(self.num_critical_nodes),
            "required_critical_passes": int(self.required_critical_passes),
            "critical_passes": int(self._critical_passes),
            "critical_seen": int(self._critical_seen),
            "remaining_critical_nodes": int(remaining_nodes),
            "remaining_required_passes": int(remaining_required),
            "risk": float(risk),
        }

    def render(self) -> str:
        return (
            f"V3[{self.difficulty or 'custom'}] step={self._step}/{self.episode_steps} "
            f"passes={self._critical_passes}/{self.required_critical_passes} "
            f"seen={self._critical_seen}/{self.num_critical_nodes} "
            f"span={self.intervention_effect_span} boost={self._boost_remaining} "
            f"critical_steps={self._critical_steps} done={self._done}"
        )
