"""Tiny result-row schema for paper experiments.

A *result row* is one JSON file under ``results/<experiment>/<run_id>.json``
holding everything needed to re-create one bar / dot in a downstream plot:

* identifying tags (experiment name, run id, seed, mode, budget, kl_coeff)
* environment config (env, horizon, episode_steps, num_actions)
* the three numbers that matter (success_rate, mean_interventions_per_traj,
  steerability metrics) parsed out of eval_success / eval_budget /
  eval_steerability JSON outputs
* artefact paths (checkpoint, raw eval reports) so re-evaluation is easy
* wall-clock + git commit for traceability

Aggregating into a single CSV is then a one-liner -- see
``experiments/aggregate_results.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


@dataclass
class ResultRow:
    schema_version: int = SCHEMA_VERSION
    experiment: str = ""
    run_id: str = ""
    seed: int = 0
    mode: str = "offline"          # "offline" | "rounds"
    budget: int = 0                # per-episode intervention budget at COLLECT
    kl_coeff: float = 0.0
    spend_mode: str = "forced"
    env: str = "v2"
    horizon: int = 0
    episode_steps: int = 0
    num_actions: int = 0
    checkpoint: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, str] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    git_commit: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def build_row(
    *,
    experiment: str,
    run_id: str,
    seed: int,
    mode: str,
    budget: int,
    kl_coeff: float,
    spend_mode: str,
    env: str,
    horizon: int,
    episode_steps: int,
    num_actions: int,
    checkpoint: str,
    eval_success_path: str,
    eval_budget_path: str,
    eval_steerability_path: str,
    wall_time_sec: float = 0.0,
    baseline_success_rate: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ResultRow:
    succ = _safe_load(eval_success_path)
    bud = _safe_load(eval_budget_path)
    steer = _safe_load(eval_steerability_path)

    metrics: Dict[str, Any] = {
        "success_rate": succ.get("success_rate"),
        "mean_reward": succ.get("mean_reward"),
        "mean_length": succ.get("mean_length"),
        "num_eval_episodes": succ.get("num_episodes"),
        "total_interventions": bud.get("total_interventions"),
        "interventions_used": bud.get("interventions_used"),
        "mean_interventions_per_traj": bud.get("mean_interventions_per_traj"),
        "mean_interventions_used": bud.get("mean_interventions_used"),
        "mean_budget_cap_per_traj": bud.get("mean_budget_cap_per_traj"),
        "budget_utilization": bud.get("budget_utilization"),
        "intervention_kind_counts": bud.get("intervention_kind_counts"),
        "avoided_bad_action_rate": steer.get("avoided_bad_action_rate"),
        "success_rate_with_intervention": steer.get(
            "success_rate_with_intervention"
        ),
        "success_rate_without_intervention": steer.get(
            "success_rate_without_intervention"
        ),
        "success_uplift_internal": steer.get("success_uplift"),
    }
    if baseline_success_rate is not None and succ.get("success_rate") is not None:
        metrics["success_uplift_vs_baseline"] = (
            float(succ["success_rate"]) - float(baseline_success_rate)
        )
        if budget > 0:
            metrics["cost_per_uplift_point"] = (
                budget / max(metrics["success_uplift_vs_baseline"] * 100.0, 1e-6)
            )

    return ResultRow(
        experiment=experiment,
        run_id=run_id,
        seed=seed,
        mode=mode,
        budget=budget,
        kl_coeff=kl_coeff,
        spend_mode=spend_mode,
        env=env,
        horizon=horizon,
        episode_steps=episode_steps,
        num_actions=num_actions,
        checkpoint=checkpoint,
        metrics=metrics,
        artefacts={
            "eval_success": eval_success_path,
            "eval_budget": eval_budget_path,
            "eval_steerability": eval_steerability_path,
        },
        wall_time_sec=wall_time_sec,
        git_commit=_git_commit(),
        extra=extra or {},
    )


def write_row(row: ResultRow, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(row), f, indent=2)
    print(f"[result_schema] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI: invoked by the shell runners after eval finishes
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge eval_success/eval_budget/eval_steerability "
                    "into a single experiment result row."
    )
    p.add_argument("--experiment", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["offline", "rounds"], default="offline")
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--kl-coeff", type=float, default=0.0)
    p.add_argument("--spend-mode", choices=["forced", "adaptive"], default="forced")
    p.add_argument("--env", default="v2")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--episode-steps", type=int, default=24)
    p.add_argument("--num-actions", type=int, default=4)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--eval-success", required=True)
    p.add_argument("--eval-budget", required=True)
    p.add_argument("--eval-steerability", required=True)
    p.add_argument("--wall-time-sec", type=float, default=0.0)
    p.add_argument("--baseline-success-rate", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--extra-json", default="",
        help="Optional JSON string merged into the row's `extra` field.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extra: Dict[str, Any] = {}
    if args.extra_json:
        extra = json.loads(args.extra_json)
    row = build_row(
        experiment=args.experiment,
        run_id=args.run_id,
        seed=args.seed,
        mode=args.mode,
        budget=args.budget,
        kl_coeff=args.kl_coeff,
        spend_mode=args.spend_mode,
        env=args.env,
        horizon=args.horizon,
        episode_steps=args.episode_steps,
        num_actions=args.num_actions,
        checkpoint=args.checkpoint,
        eval_success_path=args.eval_success,
        eval_budget_path=args.eval_budget,
        eval_steerability_path=args.eval_steerability,
        wall_time_sec=args.wall_time_sec,
        baseline_success_rate=args.baseline_success_rate,
        extra=extra,
    )
    write_row(row, args.out)
