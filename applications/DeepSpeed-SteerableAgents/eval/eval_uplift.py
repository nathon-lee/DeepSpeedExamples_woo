"""Cross-experiment steering-uplift analysis.

Combines a *baseline* `eval_success.json` report (typically a student
evaluated without any steering, i.e. ``GLOBAL_BUDGET=0``) with one or
more *steered* runs and emits an uplift table.

Each steered run is described by a pair of JSON files:

* ``<name>_success.json`` produced by ``eval/eval_success.py``
* ``<name>_budget.json``  produced by ``eval/eval_budget.py``

Per-run we report:

* ``success_rate``
* ``success_uplift``        = success_rate - baseline.success_rate
* ``num_interventions``     (sum across the evaluation rollouts)
* ``cost_per_uplift_point`` = num_interventions / (100 * success_uplift)

The script does **no** training or rollout collection of its own; it is a
pure post-hoc aggregator so it can be composed freely with a shell loop
over budgets.

Example
-------

Run the collect → train → eval pipeline once per budget, then aggregate:

```bash
mkdir -p sweep && cd sweep
NUM_EPISODES=512
for B in 0 2 4 8 16; do
    GLOBAL_BUDGET=$((B*NUM_EPISODES)) PER_EP_BUDGET=$B THRESHOLD=0 \
        ENV=v2 HORIZON=8 EPISODE_STEPS=24 NUM_EPISODES=$NUM_EPISODES \
        OUTPUT=roll_b${B}.jsonl bash ../scripts/run_collect.sh

    ENV=v2 HORIZON=8 EPISODE_STEPS=24 NUM_STEPS=2000 BATCH_SIZE=128 \
        ROLLOUTS=roll_b${B}.jsonl \
        CHECKPOINT=ckpt_b${B}.pt bash ../scripts/run_train.sh

    ENV=v2 HORIZON=8 EPISODE_STEPS=24 NUM_EVAL_EPISODES=256 \
        CHECKPOINT=ckpt_b${B}.pt ROLLOUTS=roll_b${B}.jsonl \
        EVAL_PREFIX=b${B}_ bash ../scripts/run_eval.sh
done

python ../eval/eval_uplift.py \\
    --baseline b0_eval_success.json \\
    --run b2:b2_eval_success.json:b2_eval_budget.json \\
    --run b4:b4_eval_success.json:b4_eval_budget.json \\
    --run b8:b8_eval_success.json:b8_eval_budget.json \\
    --run b16:b16_eval_success.json:b16_eval_budget.json \\
    --output uplift.json
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run_spec(spec: str) -> Tuple[str, str, str]:
    """Parse ``name:success.json:budget.json``.

    The ``name`` is a short label used in the output table; the two paths
    point at the per-run eval reports.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--run expects 'name:success.json:budget.json', got {spec!r}"
        )
    name, success_path, budget_path = parts
    if not name:
        raise ValueError(f"--run has empty name: {spec!r}")
    for p in (success_path, budget_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    return name, success_path, budget_path


def _interventions_total(budget_report: Dict[str, Any]) -> int:
    """Best-effort extraction of total intervention count.

    Supports a few common keys produced by ``eval/eval_budget.py`` so the
    aggregator stays robust to small report-format changes.
    """
    for key in ("total_interventions", "num_interventions", "interventions"):
        if key in budget_report and isinstance(budget_report[key], (int, float)):
            return int(budget_report[key])
    # Fall back to summing a per-kind dict if present.
    by_kind = budget_report.get("intervention_kind_counts") or budget_report.get(
        "interventions_by_kind"
    )
    if isinstance(by_kind, dict):
        return int(sum(v for v in by_kind.values() if isinstance(v, (int, float))))
    return 0


def aggregate(baseline_path: str, run_specs: List[str]) -> Dict[str, Any]:
    baseline = _load_json(baseline_path)
    baseline_succ = float(baseline.get("success_rate", 0.0))

    rows: List[Dict[str, Any]] = []
    for spec in run_specs:
        name, succ_path, bud_path = _parse_run_spec(spec)
        succ_report = _load_json(succ_path)
        bud_report = _load_json(bud_path)

        succ = float(succ_report.get("success_rate", 0.0))
        uplift = succ - baseline_succ
        n_iv = _interventions_total(bud_report)
        # Cost per uplift "point" = per +1% absolute success-rate gain.
        if uplift > 0:
            cpup: float | None = n_iv / (100.0 * uplift)
        else:
            cpup = None

        rows.append(
            {
                "name": name,
                "success_rate": succ,
                "success_uplift": uplift,
                "num_interventions": n_iv,
                "cost_per_uplift_point": cpup,
            }
        )

    return {
        "baseline_success_rate": baseline_succ,
        "baseline_path": baseline_path,
        "runs": rows,
    }


def _format_table(report: Dict[str, Any]) -> str:
    head = (
        f"baseline success_rate = {report['baseline_success_rate']:.3f}\n"
        f"{'name':<10} {'succ':>6} {'uplift':>8} {'n_iv':>8} {'cost/+1%':>10}"
    )
    lines = [head]
    for r in report["runs"]:
        cpup = (
            "n/a" if r["cost_per_uplift_point"] is None
            else f"{r['cost_per_uplift_point']:.2f}"
        )
        lines.append(
            f"{r['name']:<10} {r['success_rate']:>6.3f} "
            f"{r['success_uplift']:>+8.3f} {r['num_interventions']:>8d} "
            f"{cpup:>10}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to the baseline eval_success.json (unsteered student).",
    )
    p.add_argument(
        "--run",
        type=str,
        action="append",
        default=[],
        metavar="NAME:SUCCESS_JSON:BUDGET_JSON",
        help="A steered run; may be passed multiple times.",
    )
    p.add_argument("--output", type=str, default="uplift.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run:
        print("[eval_uplift] no --run specs provided; nothing to do.", file=sys.stderr)
        sys.exit(2)

    report = aggregate(args.baseline, args.run)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(_format_table(report))
    print(f"\n[eval_uplift] wrote {args.output}")


if __name__ == "__main__":
    main()
