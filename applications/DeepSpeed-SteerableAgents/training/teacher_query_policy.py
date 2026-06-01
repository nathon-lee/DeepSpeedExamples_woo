"""Teacher query policy.

For the MVP, the "teacher" is a deterministic oracle over the toy env. Given
probe info (oracle/trap/progress/critical-node flags), the teacher decides
whether to emit an ``InterventionEvent`` and what action override/logits to
attach.

This module now supports adaptive spend mode cleanly: the teacher may decline
(``None``) on low-risk steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.schema import InterventionEvent


def _peaked_logits(
    target_action: int, num_actions: int, peak: float = 2.0
) -> List[float]:
    """Build a smoothed-peaked logits vector centered on ``target_action``."""
    logits = [-peak] * int(num_actions)
    if 0 <= int(target_action) < int(num_actions):
        logits[int(target_action)] = peak
    return logits


class TeacherQueryPolicy:
    """Rule-based teacher that picks informative interventions."""

    def __init__(
        self,
        num_actions: int,
        teacher_id: str = "oracle",
        veto_peak: float = 3.0,
        plan_peak: float = 2.0,
        uncertainty_threshold: float = 0.9,
    ) -> None:
        self.num_actions = int(num_actions)
        self.teacher_id = teacher_id
        self.veto_peak = float(veto_peak)
        self.plan_peak = float(plan_peak)
        self.uncertainty_threshold = float(uncertainty_threshold)

    def query(
        self,
        step: int,
        proposed_action: int,
        info: Dict[str, Any],
        uncertainty: float,
    ) -> Optional[InterventionEvent]:
        """Return an InterventionEvent or None if the teacher declines.

        Rules:
          - If proposed action is known trap, emit action_veto.
          - Else if uncertainty is high OR this is a critical node, emit
            plan_correction with oracle action.
          - Else decline (adaptive no-spend).
        """
        oracle_action = info.get("oracle_action")
        trap_action = info.get("trap_action")
        is_critical_node = bool(info.get("is_critical_node", False))

        if oracle_action is None:
            return None

        if trap_action is not None and proposed_action == trap_action:
            return InterventionEvent(
                step=step,
                kind="action_veto",
                payload={
                    "vetoed_action": int(proposed_action),
                    "replacement_action": int(oracle_action),
                    "reason": "trap_action",
                    "quality": 2.0,
                    "teacher_logits": _peaked_logits(
                        int(oracle_action), self.num_actions, self.veto_peak
                    ),
                },
                cost=1.0,
                teacher_id=self.teacher_id,
            )

        if uncertainty >= self.uncertainty_threshold or is_critical_node:
            return InterventionEvent(
                step=step,
                kind="plan_correction",
                payload={
                    "first_action": int(oracle_action),
                    "target_action": int(oracle_action),
                    "quality": 1.5,
                    "teacher_logits": _peaked_logits(
                        int(oracle_action), self.num_actions, self.plan_peak
                    ),
                },
                cost=1.0,
                teacher_id=self.teacher_id,
            )

        return None
