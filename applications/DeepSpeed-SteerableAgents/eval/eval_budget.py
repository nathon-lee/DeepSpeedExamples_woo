"""Budget consumption evaluation over a collected rollouts file."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from data.intervention_dataset import load_trajectories_jsonl  # noqa: E402


def evaluate(rollouts_path: str) -> dict:
    trajs = load_trajectories_jsonl(rollouts_path)
    iv_counts = [len(t.interventions) for t in trajs]
    costs = [sum(iv.cost for iv in t.interventions) for t in trajs]
    cap_counts = [
        float(t.info.get("per_episode_budget_cap", 0.0)) for t in trajs
    ]
    kinds = Counter()
    for t in trajs:
        for iv in t.interventions:
            kinds[iv.kind] += 1

    total_cap = sum(cap_counts)
    total_interventions = sum(iv_counts)
    utilization = (
        total_interventions / total_cap if total_cap > 0 else None
    )

    report = {
        "num_trajectories": len(trajs),
        "total_interventions": total_interventions,
        "total_cost": sum(costs),
        "total_budget_cap": total_cap,
        "interventions_used": total_interventions,
        "mean_interventions_per_traj": (
            sum(iv_counts) / max(1, len(trajs))
        ),
        "mean_interventions_used": sum(iv_counts) / max(1, len(trajs)),
        "mean_budget_cap_per_traj": sum(cap_counts) / max(1, len(cap_counts)),
        "budget_utilization": utilization,
        "mean_cost_per_traj": sum(costs) / max(1, len(trajs)),
        "max_interventions_in_a_traj": max(iv_counts, default=0),
        "intervention_kind_counts": dict(kinds),
    }
    print(f"[eval_budget] {json.dumps(report, indent=2)}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, default="rollouts.jsonl")
    p.add_argument("--output", type=str, default="eval_budget.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = evaluate(args.rollouts)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
