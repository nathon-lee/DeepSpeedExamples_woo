"""Simple steerability metrics.

We compute:

- ``mean_interventions_per_trajectory``
- ``avoided_bad_action_rate``: fraction of ``action_veto`` events whose
  vetoed action was the trap (i.e. teacher actively prevented a failure).
- ``success_rate_with_intervention`` vs ``success_rate_without_intervention``:
  by partitioning trajectories on whether they contained at least one
  intervention.
- ``success_uplift``: difference of the two above.

These are intentionally lightweight; the goal is to give the paper-prototype
a baseline scoreboard that can be extended.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from data.intervention_dataset import load_trajectories_jsonl  # noqa: E402


def evaluate(rollouts_path: str) -> dict:
    trajs = load_trajectories_jsonl(rollouts_path)
    n = len(trajs)
    if n == 0:
        raise RuntimeError(f"No trajectories in {rollouts_path}")

    iv_counts = [len(t.interventions) for t in trajs]
    has_iv = [c > 0 for c in iv_counts]
    successes = [bool(t.success) for t in trajs]

    with_iv = [s for s, h in zip(successes, has_iv) if h]
    without_iv = [s for s, h in zip(successes, has_iv) if not h]

    veto_events = [
        iv for t in trajs for iv in t.interventions if iv.kind == "action_veto"
    ]
    trap_vetoes = sum(
        1 for iv in veto_events if iv.payload.get("reason") == "trap_action"
    )

    sr_with = sum(with_iv) / len(with_iv) if with_iv else None
    sr_without = sum(without_iv) / len(without_iv) if without_iv else None

    report = {
        "num_trajectories": n,
        "mean_interventions_per_trajectory": sum(iv_counts) / n,
        "fraction_trajectories_with_intervention": sum(has_iv) / n,
        "num_action_vetoes": len(veto_events),
        "avoided_bad_action_rate": (
            trap_vetoes / len(veto_events) if veto_events else None
        ),
        "success_rate_with_intervention": sr_with,
        "success_rate_without_intervention": sr_without,
        "success_uplift": (
            (sr_with - sr_without)
            if (sr_with is not None and sr_without is not None)
            else None
        ),
    }
    print(f"[eval_steerability] {json.dumps(report, indent=2)}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, default="rollouts.jsonl")
    p.add_argument("--output", type=str, default="eval_steerability.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = evaluate(args.rollouts)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
