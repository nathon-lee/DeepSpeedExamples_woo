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
) -> List[Trajectory]:
    torch.manual_seed(seed)

    env_kwargs: Dict[str, Any] = {}
    if env_name == "v2" and episode_steps is not None:
        env_kwargs["episode_steps"] = int(episode_steps)
    env = make_env(
        env_name, num_actions=num_actions, horizon=horizon, seed=seed,
        **env_kwargs,
    )
    student = MLPPolicy(obs_dim=env.obs_dim, num_actions=env.num_actions)
    if checkpoint and os.path.isfile(checkpoint):
        student.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    student.eval()

    teacher = TeacherQueryPolicy()
    budget = BudgetController(
        global_budget=global_budget,
        per_episode_budget=per_episode_budget,
        threshold=threshold,
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
        # state via a soft API: env exposes oracle/trap in step()'s info AFTER the
        # action. To still let the teacher veto traps, we ask the env for its
        # currently-sampled oracle/trap via a non-public probe.
        while not done and step < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, logits, entropy = student.act(obs_t, greedy=False)
            # Pre-step probe: peek at the env's hidden state for the teacher.
            probe_info = {
                "oracle_action": env._good_action,  # noqa: SLF001
                "trap_action": env._trap_action,    # noqa: SLF001
                "progress": env._progress,          # noqa: SLF001
            }

            executed_action = action
            iv: InterventionEvent | None = None
            if budget.should_intervene(entropy, step, max_steps):
                iv = teacher.query(
                    step=step,
                    proposed_action=action,
                    info=probe_info,
                    uncertainty=entropy,
                )
                if iv is not None:
                    budget.record_intervention(cost=1)
                    if iv.kind == "action_veto":
                        executed_action = int(
                            iv.payload.get("replacement_action", action)
                        )
                    elif iv.kind == "plan_correction":
                        executed_action = int(
                            iv.payload.get("first_action", action)
                        )
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
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v2"])
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="V2 only: cap on episode length; defaults to --horizon. "
             "Set larger than --horizon for a smoother budget-vs-success curve.",
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
    )
