"""Rollout collection with budgeted teacher steering.

Runs ``num_episodes`` of the toy env with a (possibly untrained) student
policy. After every action, the budget controller decides whether to query
the teacher; if it does, the teacher emits an ``InterventionEvent`` which is
attached to the trajectory. The action that is actually executed in the env
is the teacher's replacement when an ``action_veto`` fires; otherwise it is
the student's sampled action.

The output is a JSONL file of ``Trajectory`` records, consumable by both the
distillation trainer and the eval scripts.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import torch

# Allow ``python training/collect_rollouts.py`` from the example dir.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from data.intervention_dataset import dump_trajectories_jsonl  # noqa: E402
from data.schema import (  # noqa: E402
    ActionStep,
    AgentState,
    InterventionEvent,
    TaskSpec,
    Trajectory,
)
from envs.factory import make_env  # noqa: E402
from models.policy_heads import MLPPolicy  # noqa: E402
from training.budget_controller import BudgetController  # noqa: E402
from training.teacher_query_policy import TeacherQueryPolicy  # noqa: E402


def collect(
    num_episodes: int,
    horizon: int,
    num_actions: int,
    global_budget: int,
    per_episode_budget: int,
    threshold: float,
    seed: int,
    output_path: str,
    checkpoint: str = "",
    env_name: str = "v1",
    episode_steps: Optional[int] = None,
    spend_mode: str = "adaptive",
    num_critical_nodes: Optional[int] = None,
    stochasticity: Optional[float] = None,
    transition_noise: Optional[float] = None,
    required_critical_passes: Optional[int] = None,
    difficulty: Optional[str] = None,
    intervention_effect_span: Optional[int] = None,
    failure_softness: Optional[str] = None,
    risk_threshold: float = 0.0,
    min_gap_between_interventions: int = 0,
    student: Optional[MLPPolicy] = None,
) -> List[Trajectory]:
    torch.manual_seed(seed)

    env_kwargs: Dict[str, Any] = {}
    if env_name in {"v2", "v3"} and episode_steps is not None:
        env_kwargs["episode_steps"] = int(episode_steps)
    if env_name == "v3":
        # Only forward explicitly-set knobs so a named difficulty preset can
        # supply the rest (explicit kwargs override the preset).
        if difficulty is not None:
            env_kwargs["difficulty"] = str(difficulty)
        if num_critical_nodes is not None:
            env_kwargs["num_critical_nodes"] = int(num_critical_nodes)
        if stochasticity is not None:
            env_kwargs["stochasticity"] = float(stochasticity)
        if transition_noise is not None:
            env_kwargs["transition_noise"] = float(transition_noise)
        if intervention_effect_span is not None:
            env_kwargs["intervention_effect_span"] = int(intervention_effect_span)
        if failure_softness is not None:
            env_kwargs["failure_softness"] = str(failure_softness)
        if required_critical_passes is not None:
            env_kwargs["required_critical_passes"] = int(required_critical_passes)
    env = make_env(
        env_name, num_actions=num_actions, horizon=horizon, seed=seed,
        **env_kwargs,
    )
    if student is None:
        student = MLPPolicy(obs_dim=env.obs_dim, num_actions=env.num_actions)
        if checkpoint and os.path.isfile(checkpoint):
            student.load_state_dict(
                torch.load(checkpoint, map_location="cpu"), strict=True
            )
    student.eval()
    # Match the obs tensors we build below to wherever the student lives
    # (rounds mode hands us a DeepSpeed-managed module on cuda:0).
    student_device = next(student.parameters()).device

    teacher = TeacherQueryPolicy(num_actions=env.num_actions)
    budget = BudgetController(
        global_budget=global_budget,
        per_episode_budget=per_episode_budget,
        threshold=threshold,
        spend_mode=spend_mode,
        risk_threshold=risk_threshold,
        min_gap_between_interventions=min_gap_between_interventions,
    )

    trajectories: List[Trajectory] = []
    # Honour the env's own episode length (V2 may allow > horizon steps).
    max_steps = getattr(env, "episode_steps", horizon)
    for ep in range(num_episodes):
        budget.state.reset_episode()
        obs = env.reset()
        task = TaskSpec(
            task_id=uuid.uuid4().hex,
            goal="reach progress >= horizon without hitting a trap",
            horizon=horizon,
            metadata={"episode_index": ep, "episode_steps": max_steps},
        )
        traj = Trajectory(task=task)
        total_reward = 0.0
        step = 0
        done = False

        # Peek at info for the *current* step by taking a dummy reset-style probe:
        # we just call env.step after deciding; teacher needs info BEFORE the act
        # is executed, so we use a one-step lookahead by reading internal hidden
        # state via the env's public probe API.
        while not done and step < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=student_device)
            action, logits, entropy = student.act(obs_t, greedy=False)
            # Pre-step probe: ask the env for whatever hidden state the
            # teacher is allowed to see. The default BaseEnv impl returns
            # ``{}``, so a real (non-toy) env can refuse to expose anything.
            probe_info = env.get_probe_info()

            executed_action = action
            iv: InterventionEvent | None = None
            is_critical = bool(probe_info.get("is_critical_node", False))
            risk = float(probe_info.get("risk", 0.0))
            if budget.should_intervene(
                entropy,
                step,
                max_steps,
                is_critical_node=is_critical,
                risk=risk,
            ):
                iv = teacher.query(
                    step=step,
                    proposed_action=action,
                    info=probe_info,
                    uncertainty=entropy,
                )
                if iv is None and budget.spend_mode == "forced":
                    # Forced-spend compatibility path: if teacher declines,
                    # still consume budget and leave the student action intact.
                    iv = InterventionEvent(
                        step=step,
                        kind="progress_update",
                        payload={
                            "progress": float(probe_info.get("progress", 0.0)),
                            "forced_spend": True,
                        },
                        cost=1.0,
                        teacher_id="oracle",
                    )
                if iv is not None:
                    budget.record_intervention(cost=1, step=step)
                    if iv.kind == "action_veto":
                        executed_action = int(
                            iv.payload.get("replacement_action", action)
                        )
                        env.notify_intervention(step)
                    elif iv.kind == "plan_correction":
                        executed_action = int(
                            iv.payload.get("first_action", action)
                        )
                        env.notify_intervention(step)
                    # progress_update / goal_redirect / request_help do not
                    # mutate the executed action by default.

            traj.states.append(
                AgentState(step=step, observation=list(obs), info={})
            )
            traj.actions.append(
                ActionStep(
                    step=step,
                    action_id=int(executed_action),
                    logits=logits.tolist(),
                    log_prob=float(
                        torch.log_softmax(logits, dim=-1)[executed_action].item()
                    ),
                    uncertainty=float(entropy),
                )
            )
            if iv is not None:
                traj.interventions.append(iv)

            obs, reward, done, info = env.step(executed_action)
            total_reward += float(reward)
            step += 1

            if done and info.get("success"):
                traj.success = True

        traj.reward = total_reward
        traj.info["steps"] = step
        traj.info["global_budget_used"] = budget.state.global_used
        traj.info["per_episode_budget_cap"] = int(per_episode_budget)
        traj.info["interventions_used"] = int(len(traj.interventions))
        traj.info["spend_mode"] = str(spend_mode)
        if env_name == "v3":
            traj.info["num_critical_nodes"] = int(
                getattr(env, "num_critical_nodes", 0)
            )
            traj.info["required_critical_passes"] = int(
                getattr(env, "required_critical_passes", 0)
            )
            traj.info["difficulty"] = str(getattr(env, "difficulty", "") or "custom")
        trajectories.append(traj)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    dump_trajectories_jsonl(output_path, trajectories)
    n_iv = sum(len(t.interventions) for t in trajectories)
    n_succ = sum(1 for t in trajectories if t.success)
    print(
        f"[collect] wrote {len(trajectories)} trajectories to {output_path} "
        f"(success={n_succ}/{len(trajectories)}, interventions={n_iv}, "
        f"budget_used={budget.state.global_used}/{global_budget})"
    )
    return trajectories


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect steered rollouts.")
    p.add_argument("--num-episodes", type=int, default=64)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=4)
    p.add_argument("--global-budget", type=int, default=128)
    p.add_argument("--per-episode-budget", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="rollouts.jsonl")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v2", "v3"])
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="V2/V3 only: cap on episode length; defaults to --horizon. "
             "Set larger than --horizon for a smoother budget-vs-success curve.",
    )
    p.add_argument(
        "--spend-mode",
        type=str,
        default="adaptive",
        choices=["forced", "adaptive"],
        help="forced: always spend when budget remains; adaptive: spend only when triggered.",
    )
    p.add_argument("--num-critical-nodes", type=int, default=None)
    p.add_argument("--stochasticity", type=float, default=None)
    p.add_argument("--transition-noise", type=float, default=None)
    p.add_argument("--required-critical-passes", type=int, default=None)
    p.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
        help="V3 only: named difficulty preset; explicit knobs override it.",
    )
    p.add_argument("--intervention-effect-span", type=int, default=None)
    p.add_argument(
        "--failure-softness",
        type=str,
        default=None,
        choices=["high", "medium", "low"],
    )
    p.add_argument(
        "--risk-threshold",
        type=float,
        default=0.0,
        help="Adaptive only: gate interventions on uncertainty/risk >= this.",
    )
    p.add_argument(
        "--min-gap-between-interventions",
        type=int,
        default=0,
        help="Adaptive only: minimum steps between two interventions.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        num_actions=args.num_actions,
        global_budget=args.global_budget,
        per_episode_budget=args.per_episode_budget,
        threshold=args.threshold,
        seed=args.seed,
        output_path=args.output,
        checkpoint=args.checkpoint,
        env_name=args.env,
        episode_steps=args.episode_steps,
        spend_mode=args.spend_mode,
        num_critical_nodes=args.num_critical_nodes,
        stochasticity=args.stochasticity,
        transition_noise=args.transition_noise,
        required_critical_passes=args.required_critical_passes,
        difficulty=args.difficulty,
        intervention_effect_span=args.intervention_effect_span,
        failure_softness=args.failure_softness,
        risk_threshold=args.risk_threshold,
        min_gap_between_interventions=args.min_gap_between_interventions,
    )
