"""Teacher query policy.

For the MVP, the 'teacher' is a deterministic oracle over the toy env. Given
the env ``info`` dict (which exposes ``oracle_action`` and ``trap_action``),
the teacher decides which ``InterventionEvent`` to emit. A real
implementation would call a stronger model or a human-in-the-loop service.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from data.schema import InterventionEvent


class TeacherQueryPolicy:
    """Rule-based teacher that picks the most informative intervention kind."""

    def __init__(self, teacher_id: str = "oracle") -> None:
        self.teacher_id = teacher_id

    def query(
        self,
        step: int,
        proposed_action: int,
        info: Dict[str, Any],
        uncertainty: float,
    ) -> Optional[InterventionEvent]:
        """Return an InterventionEvent or None if the teacher declines.

        Rules (in order):
          - If the proposed action is the trap, **veto** it and provide the
            oracle replacement action.
          - Else if uncertainty is high, give a **plan_correction** with the
            oracle action as the first action.
          - Else periodically emit a **progress_update**.
        """
        oracle_action = info.get("oracle_action")
        trap_action = info.get("trap_action")
        progress = info.get("progress", 0)

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
                },
                cost=1.0,
                teacher_id=self.teacher_id,
            )

        if uncertainty >= 0.9:
            return InterventionEvent(
                step=step,
                kind="plan_correction",
                payload={
                    "first_action": int(oracle_action),
                    "target_action": int(oracle_action),
                    "quality": 1.5,
                },
                cost=1.0,
                teacher_id=self.teacher_id,
            )

        if step > 0 and step % 8 == 0:
            return InterventionEvent(
                step=step,
                kind="progress_update",
                payload={"progress": float(progress), "quality": 0.5},
                cost=0.5,
                teacher_id=self.teacher_id,
            )

        # Otherwise: still spend if asked, give a benign plan correction.
        return InterventionEvent(
            step=step,
            kind="plan_correction",
            payload={
                "first_action": int(oracle_action),
                "target_action": int(oracle_action),
                "quality": 1.0,
            },
            cost=1.0,
            teacher_id=self.teacher_id,
        )
