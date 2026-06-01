"""Budget controller.

Decides whether to *request* a teacher intervention this step, given
uncertainty, horizon progress, and remaining budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetState:
    """Mutable budget state shared across an episode / experiment."""
    global_budget: int
    per_episode_budget: int
    global_used: int = 0
    episode_used: int = 0

    def reset_episode(self) -> None:
        self.episode_used = 0

    @property
    def global_remaining(self) -> int:
        return max(0, self.global_budget - self.global_used)

    @property
    def episode_remaining(self) -> int:
        return max(0, self.per_episode_budget - self.episode_used)

    def can_spend(self) -> bool:
        return self.global_remaining > 0 and self.episode_remaining > 0

    def spend(self, cost: int = 1) -> None:
        self.global_used += cost
        self.episode_used += cost


class BudgetController:
    """Heuristic 'should I ask for help?' controller.

    Triggers an intervention request when

      ``score = w_u * uncertainty + w_h * horizon_pressure >= threshold``

    and budget is available. ``horizon_pressure`` grows as we get closer to the
    end of the episode, so the controller is more eager to spend late in long
    episodes (when mistakes are more expensive).
    """

    def __init__(
        self,
        global_budget: int,
        per_episode_budget: int,
        threshold: float = 0.6,
        w_uncertainty: float = 1.0,
        w_horizon: float = 0.3,
        spend_mode: str = "forced",
        w_critical: float = 0.7,
    ) -> None:
        self.state = BudgetState(
            global_budget=int(global_budget),
            per_episode_budget=int(per_episode_budget),
        )
        self.threshold = float(threshold)
        self.w_uncertainty = float(w_uncertainty)
        self.w_horizon = float(w_horizon)
        self.w_critical = float(w_critical)
        sm = (spend_mode or "forced").lower()
        if sm not in {"forced", "adaptive"}:
            raise ValueError("spend_mode must be 'forced' or 'adaptive'")
        self.spend_mode = sm

    def score(
        self,
        uncertainty: float,
        step: int,
        horizon: int,
        is_critical_node: bool = False,
    ) -> float:
        horizon_pressure = step / max(1, horizon)
        critical_bonus = self.w_critical if is_critical_node else 0.0
        return (
            self.w_uncertainty * float(uncertainty)
            + self.w_horizon * horizon_pressure
            + critical_bonus
        )

    def should_intervene(
        self,
        uncertainty: float,
        step: int,
        horizon: int,
        is_critical_node: bool = False,
    ) -> bool:
        if not self.state.can_spend():
            return False
        if self.spend_mode == "forced":
            return True
        return (
            self.score(
                uncertainty, step, horizon, is_critical_node=is_critical_node
            )
            >= self.threshold
        )

    def record_intervention(self, cost: int = 1) -> None:
        self.state.spend(cost)
