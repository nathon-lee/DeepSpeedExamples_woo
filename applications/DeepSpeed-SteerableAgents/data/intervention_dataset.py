"""Dataset wrappers over collected trajectories.

Reads a JSONL file of ``Trajectory`` records and exposes per-step
``DistillationSample`` objects. Each step that has an associated
``InterventionEvent`` becomes a labeled supervision example; steps without
intervention are skipped (the default behavior for behavior cloning from
teacher corrections).
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

from .schema import DistillationSample, InterventionEvent, Trajectory


def _target_from_intervention(
    ev: InterventionEvent, fallback_action: int
) -> Optional[int]:
    """Extract a target action from an intervention payload, if present."""
    payload = ev.payload or {}
    if "target_action" in payload:
        return int(payload["target_action"])
    if ev.kind == "action_veto" and "replacement_action" in payload:
        return int(payload["replacement_action"])
    if ev.kind == "plan_correction" and "first_action" in payload:
        return int(payload["first_action"])
    # progress_update / request_help w/o explicit target -> no supervision
    return None if ev.kind in ("progress_update", "request_help") else fallback_action


def iter_distillation_samples(traj: Trajectory) -> Iterator[DistillationSample]:
    """Yield distillation samples from a single trajectory."""
    iv_by_step = {iv.step: iv for iv in traj.interventions}
    for state, action in zip(traj.states, traj.actions):
        ev = iv_by_step.get(state.step)
        if ev is None:
            continue
        target = _target_from_intervention(ev, fallback_action=action.action_id)
        if target is None:
            continue
        quality = float(ev.payload.get("quality", 1.0))
        yield DistillationSample(
            observation=list(state.observation),
            target_action=int(target),
            teacher_logits=ev.payload.get("teacher_logits"),
            student_logits=action.logits,
            quality=quality,
            source_event_id=ev.event_id,
            source_task_id=traj.task.task_id,
        )


def load_trajectories_jsonl(path: str) -> List[Trajectory]:
    out: List[Trajectory] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.from_json(line))
    return out


def load_distillation_samples(path: str) -> List[DistillationSample]:
    samples: List[DistillationSample] = []
    for traj in load_trajectories_jsonl(path):
        samples.extend(iter_distillation_samples(traj))
    return samples


def dump_trajectories_jsonl(path: str, trajectories: List[Trajectory]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(t.to_json() + "\n")
