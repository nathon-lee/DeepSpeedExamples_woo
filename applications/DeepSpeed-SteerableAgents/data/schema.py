"""Data schema for trajectories and human/teacher interventions.

The schema is intentionally lightweight (Python dataclasses) so it can be
serialized to JSONL and reloaded without any framework dependency.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

InterventionKind = str  # one of INTERVENTION_KINDS below

INTERVENTION_KINDS = (
    "progress_update",   # teacher reports current task progress / status
    "request_help",      # agent asks teacher; teacher answers
    "plan_correction",   # teacher rewrites the agent's plan
    "action_veto",       # teacher vetoes the proposed action
    "goal_redirect",     # teacher changes / refines the task goal
)


@dataclass
class TaskSpec:
    """High-level description of a long-horizon task."""
    task_id: str
    goal: str
    horizon: int                              # max number of agent steps
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """A snapshot of the agent / environment observation at a given step."""
    step: int
    observation: List[float]                  # numeric feature vector
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionStep:
    """An action taken by the student policy at a given step."""
    step: int
    action_id: int
    logits: Optional[List[float]] = None      # student logits (for distillation)
    log_prob: Optional[float] = None
    uncertainty: Optional[float] = None       # e.g. predictive entropy


@dataclass
class ProgressUpdate:
    """Structured progress signal (used inside InterventionEvent payloads)."""
    step: int
    progress: float                            # in [0, 1]
    note: str = ""


@dataclass
class InterventionEvent:
    """A single steering event from a teacher / human."""
    step: int
    kind: InterventionKind
    payload: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    teacher_id: str = "oracle"
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise ValueError(
                f"InterventionEvent.kind={self.kind!r} not in {INTERVENTION_KINDS}"
            )


@dataclass
class Trajectory:
    """A full episode: task + per-step states/actions + interventions."""
    task: TaskSpec
    states: List[AgentState] = field(default_factory=list)
    actions: List[ActionStep] = field(default_factory=list)
    interventions: List[InterventionEvent] = field(default_factory=list)
    success: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Trajectory":
        d = json.loads(s)
        return cls(
            task=TaskSpec(**d["task"]),
            states=[AgentState(**x) for x in d["states"]],
            actions=[ActionStep(**x) for x in d["actions"]],
            interventions=[InterventionEvent(**x) for x in d["interventions"]],
            success=d.get("success", False),
            reward=d.get("reward", 0.0),
            info=d.get("info", {}),
        )


@dataclass
class DistillationSample:
    """A single (observation, target) pair extracted from intervention data."""
    observation: List[float]
    target_action: int
    teacher_logits: Optional[List[float]] = None
    student_logits: Optional[List[float]] = None
    quality: float = 1.0                       # for prioritized replay
    source_event_id: Optional[str] = None
    source_task_id: Optional[str] = None
