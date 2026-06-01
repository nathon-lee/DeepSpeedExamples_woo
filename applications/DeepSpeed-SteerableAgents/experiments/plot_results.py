"""Generate the four headline plots from ``results/all.csv``.

Pure reporting script — does not import or modify any training code.

Outputs (under ``--out-dir`` / ``plots/`` by default):
    success_vs_budget.png       success vs budget, mean ± std (5 seeds)
    cost_per_uplift.png         cost / uplift point vs budget
    distill_ablation.png        BC-only vs BC+KL bar with std error bars
    offline_vs_rounds.png       offline vs rounds bar with std error bars

Usage:
    python experiments/plot_results.py                # reads results/all.csv
    python experiments/plot_results.py --csv path/all.csv --out-dir plots/
    python experiments/plot_results.py --env v3 --spend-mode adaptive
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(x: str):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _group(rows: List[Dict[str, str]],
           key_cols: List[str]) -> Dict[Tuple, List[Dict[str, str]]]:
    buckets: Dict[Tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(c, "") for c in key_cols)].append(r)
    return buckets


def _agg(rows: List[Dict[str, str]], metric: str) -> Tuple[float, float, int]:
    vals = [v for v in (_as_float(r.get(metric)) for r in rows) if v is not None]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n >= 2 else 0.0
    return mean, std, n


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}", file=sys.stderr)
        print("[plot] install with: pip install matplotlib", file=sys.stderr)
        sys.exit(2)


def _filtered(rows, env_name: str, spend_mode: str):
    return [
        r for r in rows
        if (not env_name or r.get("env") == env_name)
        and (not spend_mode or r.get("spend_mode") == spend_mode)
    ]


def plot_budget_sweep(rows, out_dir: str, env_name: str, spend_mode: str) -> None:
    import matplotlib.pyplot as plt
    bs = [
        r for r in _filtered(rows, env_name, spend_mode)
        if r.get("experiment") == "budget_sweep"
    ]
    if not bs:
        print("[plot] no budget_sweep rows; skipping", file=sys.stderr); return
    buckets = _group(bs, ["budget"])
    xs, ys, errs = [], [], []
    for budget, members in sorted(buckets.items(),
                                  key=lambda kv: float(kv[0][0])):
        m, s, n = _agg(members, "success_rate")
        xs.append(float(budget[0])); ys.append(m * 100); errs.append(s * 100)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, linewidth=2,
                color="#1f77b4", ecolor="#888")
    ax.set_xlabel("Per-episode intervention budget B")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        f"Success vs. intervention budget (env={env_name}, spend={spend_mode})"
    )
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    # Annotate peak.
    peak_idx = max(range(len(ys)), key=lambda i: ys[i])
    ax.annotate(f"peak {ys[peak_idx]:.1f}%",
                xy=(xs[peak_idx], ys[peak_idx]),
                xytext=(xs[peak_idx] + 0.7, ys[peak_idx] - 8),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
    fig.tight_layout()
    out = os.path.join(out_dir, "success_vs_budget.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_cost_per_uplift(rows, out_dir: str, env_name: str, spend_mode: str) -> None:
    import matplotlib.pyplot as plt
    bs = [r for r in _filtered(rows, env_name, spend_mode) if r.get("experiment") == "budget_sweep"
          and _as_float(r.get("cost_per_uplift_point")) is not None]
    if not bs:
        return
    buckets = _group(bs, ["budget"])
    xs, ys, errs = [], [], []
    for budget, members in sorted(buckets.items(),
                                  key=lambda kv: float(kv[0][0])):
        m, s, _ = _agg(members, "cost_per_uplift_point")
        xs.append(float(budget[0])); ys.append(m); errs.append(s)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(xs, ys, yerr=errs, marker="s", capsize=4, linewidth=2,
                color="#d62728", ecolor="#888")
    ax.set_xlabel("Per-episode intervention budget B")
    ax.set_ylabel("Cost per uplift point  (budget / uplift%)")
    ax.set_title("Intervention efficiency (lower = better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "cost_per_uplift.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_ablation(rows, out_dir: str, env_name: str, spend_mode: str) -> None:
    import matplotlib.pyplot as plt
    rs = [
        r for r in _filtered(rows, env_name, spend_mode)
        if r.get("experiment") == "distill_ablation"
    ]
    if not rs:
        return
    buckets = _group(rs, ["kl_coeff"])
    keys = sorted(buckets.keys(), key=lambda k: float(k[0]))
    labels, means, stds = [], [], []
    for k in keys:
        m, s, _ = _agg(buckets[k], "success_rate")
        kl = float(k[0])
        labels.append(f"BC-only\n(kl={kl})" if kl == 0 else f"BC + KL\n(kl={kl})")
        means.append(m * 100); stds.append(s * 100)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=["#2ca02c", "#ff7f0e"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        f"Distillation ablation @ B=4 (env={env_name}, spend={spend_mode})"
    )
    ax.set_ylim(min(means) - 4, 100.5)
    ax.grid(True, axis="y", alpha=0.3)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.2,
                f"{m:.1f}%", ha="center", fontsize=10, fontweight="bold")
    if len(means) == 2:
        delta = means[0] - means[1]
        ax.set_xlabel(f"Δ = {delta:+.1f} pp  (BC-only − BC+KL)")
    fig.tight_layout()
    out = os.path.join(out_dir, "distill_ablation.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_offline_vs_rounds(rows, out_dir: str, env_name: str, spend_mode: str) -> None:
    import matplotlib.pyplot as plt
    rs = [
        r for r in _filtered(rows, env_name, spend_mode)
        if r.get("experiment") == "offline_vs_rounds"
    ]
    if not rs:
        return
    buckets = _group(rs, ["mode"])
    order = ["offline", "rounds"]
    labels, means, stds = [], [], []
    for mode in order:
        members = buckets.get((mode,), [])
        if not members:
            continue
        m, s, _ = _agg(members, "success_rate")
        labels.append(mode); means.append(m * 100); stds.append(s * 100)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=["#1f77b4", "#9467bd"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(
        "Offline vs. rounds @ B=4, kl=0.0 "
        f"(env={env_name}, spend={spend_mode})"
    )
    ax.set_ylim(min(means) - 4, 100.5)
    ax.grid(True, axis="y", alpha=0.3)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.2,
                f"{m:.1f}%", ha="center", fontsize=10, fontweight="bold")
    if len(means) == 2:
        delta = means[0] - means[1]
        ax.set_xlabel(f"Δ = {delta:+.1f} pp  (offline − rounds)")
    fig.tight_layout()
    out = os.path.join(out_dir, "offline_vs_rounds.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/all.csv",
                   help="Flat CSV produced by aggregate_results.py")
    p.add_argument("--out-dir", default="plots",
                   help="Where to write the .png files")
    p.add_argument("--env", default="v3",
                   help="Filter rows by env value before plotting")
    p.add_argument("--spend-mode", default="adaptive",
                   help="Filter rows by spend_mode before plotting")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[plot] input CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    _ensure_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = _read_csv(args.csv)
    print(f"[plot] loaded {len(rows)} rows from {args.csv}")

    plot_budget_sweep(rows, args.out_dir, args.env, args.spend_mode)
    plot_cost_per_uplift(rows, args.out_dir, args.env, args.spend_mode)
    plot_ablation(rows, args.out_dir, args.env, args.spend_mode)
    plot_offline_vs_rounds(rows, args.out_dir, args.env, args.spend_mode)


if __name__ == "__main__":
    main()
