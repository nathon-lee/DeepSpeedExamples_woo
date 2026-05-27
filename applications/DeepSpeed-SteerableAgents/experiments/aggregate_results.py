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
    # Dedup by (experiment, run_id). Same logical run scanned via two paths
    # (e.g. ``results/`` and ``results/<exp>/``) must collapse to one row,
    # otherwise downstream std / counts are silently wrong.
    by_key: Dict[tuple, Dict[str, Any]] = {}
    dup_count = 0
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
        key = (row.get("experiment"), row.get("run_id"))
        if key in by_key:
            dup_count += 1
            continue
        by_key[key] = row
    if dup_count:
        print(f"[aggregate] dropped {dup_count} duplicate row(s) by (experiment, run_id)",
              file=sys.stderr)
    return list(by_key.values())


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


def _group_summary(rows: List[Dict[str, Any]],
                   group_cols: List[str],
                   metric_cols: List[str]) -> List[Dict[str, Any]]:
    """Group rows by ``group_cols`` and compute mean/std/n per metric.

    Returns a list of flat dicts with keys ``group_cols + [<metric>_mean,
    <metric>_std, <metric>_n for metric in metric_cols]``.
    """
    import math
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(c) for c in group_cols)
        buckets.setdefault(key, []).append(r)
    out: List[Dict[str, Any]] = []
    for key, members in sorted(buckets.items(), key=lambda kv: tuple(
            "" if v is None else str(v) for v in kv[0])):
        agg: Dict[str, Any] = {c: v for c, v in zip(group_cols, key)}
        for m in metric_cols:
            vals = [r.get(m) for r in members if isinstance(
                r.get(m), (int, float)) and not (
                isinstance(r.get(m), float) and math.isnan(r.get(m)))]
            n = len(vals)
            agg[f"{m}_n"] = n
            if n == 0:
                agg[f"{m}_mean"] = None
                agg[f"{m}_std"] = None
                continue
            mean = sum(vals) / n
            if n >= 2:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            agg[f"{m}_mean"] = round(mean, 6)
            agg[f"{m}_std"] = round(std, 6)
        out.append(agg)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Files or directories to scan")
    p.add_argument("--csv", default="", help="Path to write CSV summary")
    p.add_argument("--markdown", default="",
                   help="Path to write a Markdown table summary")
    p.add_argument(
        "--group-by", default="",
        help="Comma-separated list of columns to group on (e.g. "
             "'experiment,mode,budget,kl_coeff'). When set, the CSV and "
             "Markdown outputs contain one row per group with mean / std / n "
             "of the metric columns.",
    )
    p.add_argument(
        "--group-metrics", default="success_rate,mean_interventions_per_traj",
        help="Comma-separated metrics to aggregate when --group-by is set.",
    )
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

    grouped_mode = bool(args.group_by)
    if grouped_mode:
        group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
        metric_cols = [c.strip() for c in args.group_metrics.split(",")
                       if c.strip()]
        rows = _group_summary(rows, group_cols, metric_cols)
        # Build column order: group cols, then mean/std/n triples per metric.
        cols: List[str] = list(group_cols)
        for m in metric_cols:
            cols.extend([f"{m}_mean", f"{m}_std", f"{m}_n"])
    else:
        # Union of keys, preserving CORE_COLS + METRIC_COLS first.
        seen = set()
        cols = []
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
        if grouped_mode:
            md_cols = cols
        else:
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
