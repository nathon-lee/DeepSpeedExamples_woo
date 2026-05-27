"""Teacher query policy.

For the MVP, the 'teacher' is a deterministic oracle over the toy env. Given
the env probe info (``oracle_action``/``trap_action``/``progress``), the
teacher decides which ``InterventionEvent`` to emit. A real implementation
would call a stronger model or a human-in-the-loop service.

The teacher additionally attaches a *smoothed peaked* ``teacher_logits``
vector to every action-bearing intervention payload so the distillation
trainer can compute a real KL term between student and teacher
distributions (not just behavior cloning on the argmax).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.schema import InterventionEvent


def _peaked_logits(
    target_action: int, num_actions: int, peak: float = 2.0
) -> List[float]:
    """Build a smoothed-peaked logits vector centered on ``target_action``.

    With ``peak=2.0`` and ``num_actions=4`` the softmax is
    ``[0.94, 0.02, 0.02, 0.02]`` -- enough mass on the recommended action to
    drive the KL term, but soft enough that the student is not forced to a
    hard one-hot.
    """
    logits = [-peak] * int(num_actions)
    if 0 <= int(target_action) < int(num_actions):
        logits[int(target_action)] = peak
    return logits


class TeacherQueryPolicy:
    """Rule-based teacher that picks the most informative intervention kind."""

    def __init__(
        self,
        num_actions: int,
        teacher_id: str = "oracle",
        veto_peak: float = 3.0,
        plan_peak: float = 2.0,
    ) -> None:
        self.num_actions = int(num_actions)
        self.teacher_id = teacher_id
        self.veto_peak = float(veto_peak)
        self.plan_peak = float(plan_peak)

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
            oracle replacement action (sharply peaked logits).
          - Else if uncertainty is high, give a **plan_correction** with the
            oracle action and a moderately peaked logits vector.
          - Else periodically emit a **progress_update** (no logits, no
            action override -- used only as a steerability signal).
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
                    "teacher_logits": _peaked_logits(
                        int(oracle_action), self.num_actions, self.veto_peak
                    ),
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
                    "teacher_logits": _peaked_logits(
                        int(oracle_action), self.num_actions, self.plan_peak
                    ),
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
                "teacher_logits": _peaked_logits(
                    int(oracle_action), self.num_actions, self.plan_peak
                ),
            },
            cost=1.0,
            teacher_id=self.teacher_id,
        )
