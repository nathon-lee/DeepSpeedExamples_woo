"""Walk ``results/`` and emit a flat CSV / Markdown summary.

Usage:
    python experiments/aggregate_results.py results/budget_sweep \
        --csv results/budget_sweep.csv

Each input directory is recursively scanned for ``*.json`` files matching the
``result_schema.ResultRow`` shape. Unknown / malformed files are skipped with
a warning instead of crashing the aggregation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List

# Columns we always emit (in this order) when present. Anything else inside
# `metrics` / `extra` is appended at the end so we never silently drop data.
CORE_COLS = [
    "experiment", "run_id", "seed", "mode", "budget", "kl_coeff",
    "env", "horizon", "episode_steps", "num_actions", "checkpoint",
    "git_commit", "wall_time_sec",
]
METRIC_COLS = [
    "success_rate", "mean_reward", "mean_length", "num_eval_episodes",
    "total_interventions", "mean_interventions_per_traj",
    "avoided_bad_action_rate",
    "success_rate_with_intervention", "success_rate_without_intervention",
    "success_uplift_internal", "success_uplift_vs_baseline",
    "cost_per_uplift_point",
]


def _load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception as exc:
            print(f"[aggregate] skip {p}: {exc}", file=sys.stderr)
            continue
        if not isinstance(row, dict) or "experiment" not in row:
            print(f"[aggregate] skip {p}: not a result row", file=sys.stderr)
            continue
        rows.append(row)
    return rows


def _flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in CORE_COLS:
        out[k] = row.get(k)
    metrics = row.get("metrics") or {}
    for k in METRIC_COLS:
        out[k] = metrics.get(k)
    extra = row.get("extra") or {}
    for k, v in extra.items():
        out[f"extra.{k}"] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Files or directories to scan")
    p.add_argument("--csv", default="", help="Path to write CSV summary")
    p.add_argument("--markdown", default="",
                   help="Path to write a Markdown table summary")
    args = p.parse_args()

    paths: List[str] = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "**", "*.json"),
                                          recursive=True)))
        else:
            paths.append(inp)

    rows = [_flatten(r) for r in _load_rows(paths)]
    if not rows:
        print("[aggregate] no result rows found", file=sys.stderr)
        sys.exit(1)

    # Union of keys, preserving CORE_COLS + METRIC_COLS first.
    seen = set()
    cols: List[str] = []
    for k in CORE_COLS + METRIC_COLS:
        cols.append(k); seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                cols.append(k); seen.add(k)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".",
                    exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        print(f"[aggregate] wrote {args.csv}  ({len(rows)} rows)")

    if args.markdown:
        # Trim to a readable subset for terminal-friendly tables.
        md_cols = [c for c in (
            "experiment", "run_id", "mode", "budget", "kl_coeff",
            "success_rate", "mean_interventions_per_traj",
            "success_uplift_vs_baseline", "cost_per_uplift_point",
        ) if c in cols]
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".",
                    exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(md_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(md_cols)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(
                    "" if r.get(k) is None else str(r.get(k))
                    for k in md_cols
                ) + " |\n")
        print(f"[aggregate] wrote {args.markdown}  ({len(rows)} rows)")

    if not args.csv and not args.markdown:
        # Print a compact summary to stdout.
        for r in rows:
            print(
                f"{r.get('experiment')}/{r.get('run_id')}  "
                f"mode={r.get('mode')} B={r.get('budget')} "
                f"kl={r.get('kl_coeff')}  "
                f"succ={r.get('success_rate')}  "
                f"int/traj={r.get('mean_interventions_per_traj')}"
            )


if __name__ == "__main__":
    main()
