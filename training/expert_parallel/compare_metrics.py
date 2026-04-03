"""Offline comparison of AutoEP vs ZeRO-3 leaf training metrics.

Reads CSV outputs from both modes and generates plots + summary JSON.

Run as a regular Python script (NOT via deepspeed launcher):
    python compare_metrics.py --autoep_csv metrics_autoep.csv --zero3_leaf_csv metrics_zero3_leaf.csv \
        --autoep_metadata run_metadata_autoep.json --zero3_leaf_metadata run_metadata_zero3_leaf.json \
        --out_dir results/ --out_json results/summary.json
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AutoEP and ZeRO-3 leaf training metrics"
    )
    parser.add_argument("--autoep_csv", type=str, required=True)
    parser.add_argument("--zero3_leaf_csv", type=str, required=True)
    parser.add_argument("--autoep_metadata", type=str, required=True)
    parser.add_argument("--zero3_leaf_metadata", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--autoep_label", type=str, default="AutoEP + ZeRO-1"
    )
    parser.add_argument(
        "--zero3_leaf_label", type=str, default="HF + ZeRO-3 leaf"
    )
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_mean_abs_diff", type=float, default=None)
    parser.add_argument("--min_post_warmup_steps", type=int, default=10)
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--require_same_init_hash",
        dest="require_same_init_hash",
        action="store_true",
        help="Require matching non-empty init_weights_sha256 in both metadata files",
    )
    hash_group.add_argument(
        "--no_require_same_init_hash",
        dest="require_same_init_hash",
        action="store_false",
        help="Allow comparison without enforcing matching init_weights_sha256",
    )
    parser.set_defaults(require_same_init_hash=True)
    return parser.parse_args()


def load_csv(path: str) -> list[dict]:
    """Load CSV and return list of row dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_metadata(path: str) -> dict:
    """Load metadata JSON."""
    with open(path) as f:
        return json.load(f)


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def validate_compatibility(
    autoep_meta: dict, zero3_meta: dict, require_same_init_hash: bool
) -> tuple[bool, list[str], list[str], bool | None, str | None, str | None]:
    """Check if runs are comparable."""
    issues = []
    warnings = []

    # Check num_layers
    a_layers = autoep_meta.get("args", {}).get("num_layers")
    z_layers = zero3_meta.get("args", {}).get("num_layers")
    if a_layers != z_layers:
        issues.append(f"num_layers mismatch: autoep={a_layers}, zero3={z_layers}")

    # Check seq_len
    a_seq = autoep_meta.get("args", {}).get("seq_len")
    z_seq = zero3_meta.get("args", {}).get("seq_len")
    if a_seq != z_seq:
        issues.append(f"seq_len mismatch: autoep={a_seq}, zero3={z_seq}")

    # Check effective tokens per update
    a_tokens = autoep_meta.get("effective_tokens_per_update")
    z_tokens = zero3_meta.get("effective_tokens_per_update")
    if a_tokens != z_tokens:
        issues.append(
            f"effective_tokens_per_update mismatch: autoep={a_tokens}, zero3={z_tokens}"
        )

    # Check precision
    a_args = autoep_meta.get("args", {})
    z_args = zero3_meta.get("args", {})

    # Check world_size
    a_ws = autoep_meta.get("world_size")
    z_ws = zero3_meta.get("world_size")
    if a_ws != z_ws:
        issues.append(f"world_size mismatch: autoep={a_ws}, zero3={z_ws}")

    a_init_hash = autoep_meta.get("init_weights_sha256")
    z_init_hash = zero3_meta.get("init_weights_sha256")
    same_init_hash = (
        a_init_hash == z_init_hash if a_init_hash and z_init_hash else None
    )

    if require_same_init_hash:
        if not a_init_hash or not z_init_hash:
            issues.append(
                "init_weights_sha256 missing in one or both metadata files "
                "(required by --require_same_init_hash)."
            )
        elif a_init_hash != z_init_hash:
            issues.append(
                f"init_weights_sha256 mismatch: autoep={a_init_hash}, zero3={z_init_hash}"
            )
    else:
        if not a_init_hash or not z_init_hash:
            warnings.append(
                "init_weights_sha256 missing in one or both metadata files; "
                "init-weight provenance is not verified."
            )
        elif a_init_hash != z_init_hash:
            warnings.append(
                "init_weights_sha256 mismatch detected but allowed by "
                "--no_require_same_init_hash."
            )

    return (
        len(issues) == 0,
        issues,
        warnings,
        same_init_hash,
        a_init_hash,
        z_init_hash,
    )


def try_plot(
    autoep_rows: list[dict],
    zero3_rows: list[dict],
    out_dir: str,
    autoep_label: str,
    zero3_leaf_label: str,
) -> dict[str, str | None]:
    """Generate comparison plots. Returns dict of plot paths or None."""
    plots = {"loss_curve": None, "peak_memory_bar": None, "throughput_bar": None}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping plots.")
        return plots

    os.makedirs(out_dir, exist_ok=True)

    # Loss curve
    try:
        a_steps = [int(r["step"]) for r in autoep_rows]
        a_loss = [float(r["loss_ce"]) for r in autoep_rows]
        z_steps = [int(r["step"]) for r in zero3_rows]
        z_loss = [float(r["loss_ce"]) for r in zero3_rows]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(a_steps, a_loss, label=autoep_label, marker="o", markersize=3)
        ax.plot(z_steps, z_loss, label=zero3_leaf_label, marker="s", markersize=3)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("CE Loss")
        ax.set_title("Loss Curve Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, "loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["loss_curve"] = path
    except Exception as e:
        print(f"WARNING: Loss curve plot failed: {e}")

    # Peak memory bar
    try:
        a_peak = max(int(r["cuda_peak_memory_allocated_bytes"]) for r in autoep_rows)
        z_peak = max(int(r["cuda_peak_memory_allocated_bytes"]) for r in zero3_rows)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [autoep_label, zero3_leaf_label],
            [a_peak / 1e9, z_peak / 1e9],
            color=["#2196F3", "#FF9800"],
        )
        ax.set_ylabel("Peak Memory (GB)")
        ax.set_title("Peak GPU Memory Comparison")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )
        path = os.path.join(out_dir, "peak_memory_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["peak_memory_bar"] = path
    except Exception as e:
        print(f"WARNING: Peak memory plot failed: {e}")

    # Throughput bar
    try:
        a_tps = [float(r["global_tokens_per_sec"]) for r in autoep_rows]
        z_tps = [float(r["global_tokens_per_sec"]) for r in zero3_rows]
        a_avg = sum(a_tps) / len(a_tps) if a_tps else 0
        z_avg = sum(z_tps) / len(z_tps) if z_tps else 0

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            [autoep_label, zero3_leaf_label],
            [a_avg, z_avg],
            color=["#2196F3", "#FF9800"],
        )
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Average Throughput Comparison (post-warmup)")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )
        path = os.path.join(out_dir, "throughput_bar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots["throughput_bar"] = path
    except Exception as e:
        print(f"WARNING: Throughput plot failed: {e}")

    return plots


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    autoep_rows = load_csv(args.autoep_csv)
    zero3_rows = load_csv(args.zero3_leaf_csv)
    autoep_meta = load_metadata(args.autoep_metadata)
    zero3_meta = load_metadata(args.zero3_leaf_metadata)

    # Validate compatibility
    (
        compatible,
        compat_issues,
        compat_warnings,
        same_init_hash,
        init_hash_autoep,
        init_hash_zero3_leaf,
    ) = validate_compatibility(
        autoep_meta,
        zero3_meta,
        require_same_init_hash=args.require_same_init_hash,
    )

    # Filter by warmup steps
    autoep_rows = [r for r in autoep_rows if int(r["step"]) >= args.warmup_steps]
    zero3_rows = [r for r in zero3_rows if int(r["step"]) >= args.warmup_steps]

    # Align steps
    autoep_steps = {int(r["step"]): r for r in autoep_rows}
    zero3_steps = {int(r["step"]): r for r in zero3_rows}
    aligned_steps = sorted(set(autoep_steps.keys()) & set(zero3_steps.keys()))

    num_aligned = len(aligned_steps)
    sufficient_evidence = num_aligned >= args.min_post_warmup_steps

    # Compute loss parity metrics
    if aligned_steps:
        a_losses = [float(autoep_steps[s]["loss_ce"]) for s in aligned_steps]
        z_losses = [float(zero3_steps[s]["loss_ce"]) for s in aligned_steps]
        abs_diffs = [abs(a - z) for a, z in zip(a_losses, z_losses)]
        mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
        max_abs_diff = max(abs_diffs)
        corr = pearson_correlation(a_losses, z_losses) if sufficient_evidence else None
    else:
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")
        corr = None

    # Check loss objective tag compatibility
    objective_mismatch = False
    if aligned_steps:
        a_tags = {autoep_steps[s].get("loss_objective_tag", "") for s in aligned_steps}
        z_tags = {zero3_steps[s].get("loss_objective_tag", "") for s in aligned_steps}
        if a_tags != z_tags:
            objective_mismatch = True

    # Threshold checks
    threshold_passed = None
    threshold_skipped_reason = None
    if args.max_mean_abs_diff is not None:
        if not compatible:
            threshold_passed = None
            threshold_skipped_reason = "Runs not comparable: " + "; ".join(compat_issues)
        elif objective_mismatch:
            threshold_passed = None
            threshold_skipped_reason = (
                "Loss objective tag mismatch between modes; threshold check skipped."
            )
        elif not sufficient_evidence:
            threshold_passed = None
            threshold_skipped_reason = (
                f"Insufficient aligned post-warmup steps ({num_aligned} < {args.min_post_warmup_steps})"
            )
        else:
            threshold_passed = mean_abs_diff <= args.max_mean_abs_diff

    # Peak memory
    a_peak_mem = (
        max(int(r["cuda_peak_memory_allocated_bytes"]) for r in autoep_rows)
        if autoep_rows
        else 0
    )
    z_peak_mem = (
        max(int(r["cuda_peak_memory_allocated_bytes"]) for r in zero3_rows)
        if zero3_rows
        else 0
    )
    mem_ratio = a_peak_mem / z_peak_mem if z_peak_mem > 0 else float("nan")

    # Throughput
    a_tps_vals = [float(r["global_tokens_per_sec"]) for r in autoep_rows]
    z_tps_vals = [float(r["global_tokens_per_sec"]) for r in zero3_rows]
    a_avg_tps = sum(a_tps_vals) / len(a_tps_vals) if a_tps_vals else 0
    z_avg_tps = sum(z_tps_vals) / len(z_tps_vals) if z_tps_vals else 0
    tps_ratio = a_avg_tps / z_avg_tps if z_avg_tps > 0 else float("nan")

    # Generate plots
    plots = try_plot(
        autoep_rows,
        zero3_rows,
        args.out_dir,
        autoep_label=args.autoep_label,
        zero3_leaf_label=args.zero3_leaf_label,
    )

    # Build summary
    summary = {
        "compatible": compatible,
        "compatibility_issues": compat_issues,
        "compatibility_warnings": compat_warnings,
        "same_init_hash": same_init_hash,
        "init_hash_autoep": init_hash_autoep,
        "init_hash_zero3_leaf": init_hash_zero3_leaf,
        "init_hash_required": args.require_same_init_hash,
        "loss_parity": {
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "pearson_correlation": corr,
            "num_aligned_steps": num_aligned,
            "num_post_warmup_steps": num_aligned,
            "sufficient_evidence": sufficient_evidence,
        },
        "threshold_checks": {
            "max_mean_abs_diff": args.max_mean_abs_diff,
            "passed": threshold_passed,
            "skipped_reason": threshold_skipped_reason,
        },
        "peak_memory": {
            "autoep_bytes": a_peak_mem,
            "zero3_leaf_bytes": z_peak_mem,
            "ratio": mem_ratio,
        },
        "throughput": {
            "autoep_tokens_per_sec": a_avg_tps,
            "zero3_leaf_tokens_per_sec": z_avg_tps,
            "ratio": tps_ratio,
        },
        "caveats": [
            "Throughput and memory comparisons include differing ZeRO stages "
            "and are not an isolated AutoEP-only benchmark.",
            "Loss comparison uses trend agreement, not bit-identical values. "
            "Small divergence is expected from different ZeRO stages and FP reduction order.",
        ],
        "autoep_metadata": autoep_meta,
        "zero3_leaf_metadata": zero3_meta,
        "plots": plots,
    }

    # Handle NaN for JSON serialization
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    summary = sanitize(summary)

    # Write summary JSON atomically
    tmp = args.out_json + ".tmp"
    parent = os.path.dirname(args.out_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_json)

    # Print summary
    print("\n=== Comparison Summary ===")
    print(f"Compatible: {compatible}")
    if compat_issues:
        print(f"Issues: {compat_issues}")
    if compat_warnings:
        print(f"Warnings: {compat_warnings}")
    print(f"Init hash required: {args.require_same_init_hash}")
    print(f"Same init hash: {same_init_hash}")
    print(f"Aligned steps: {num_aligned}")
    print(f"Mean abs diff (loss): {mean_abs_diff}")
    print(f"Max abs diff (loss): {max_abs_diff}")
    if corr is not None:
        print(f"Pearson correlation: {corr:.4f}")
    print(f"Peak memory ratio (autoep/zero3): {mem_ratio}")
    print(f"Throughput ratio (autoep/zero3): {tps_ratio}")
    if threshold_passed is not None:
        print(f"Threshold check: {'PASSED' if threshold_passed else 'FAILED'}")
    print(f"\nSummary written to: {args.out_json}")

    if plots["loss_curve"]:
        print(f"Loss curve plot: {plots['loss_curve']}")
    if plots["peak_memory_bar"]:
        print(f"Memory plot: {plots['peak_memory_bar']}")
    if plots["throughput_bar"]:
        print(f"Throughput plot: {plots['throughput_bar']}")

    # Exit with non-zero for compatibility failures or threshold failure.
    if not compatible or threshold_passed is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
