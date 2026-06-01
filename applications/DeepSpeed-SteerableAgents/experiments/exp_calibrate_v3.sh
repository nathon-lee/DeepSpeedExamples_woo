#!/usr/bin/env bash
# Calibration sweep for the V3 environment.
#
# Goal: find the difficulty preset whose budgeted-steering curve lands in a
# discriminative mid-range (B=0 low, rising with budget, B=8 high-but-not-100%).
#
# Runs env=v3, spend_mode=adaptive, budgets {0,1,2,4,8}, seed 0 by default,
# across the difficulty presets {easy, medium, hard}, then aggregates a single
# comparison table grouped by (difficulty, budget).
#
# Override DIFFICULTIES / BUDGETS / SEEDS / RESULTS_DIR via env vars.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"

DIFFICULTIES="${DIFFICULTIES:-easy medium hard}"
SEEDS="${SEEDS:-0}"
BUDGETS="${BUDGETS:-0 1 2 4 8}"
ENV="v3"
SPEND_MODE="${SPEND_MODE:-adaptive}"

# Lighter defaults than the full multi-seed sweep so a calibration pass is fast.
NUM_EPISODES="${NUM_EPISODES:-256}"
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-256}"
THRESHOLD="${THRESHOLD:-0.95}"
KL_COEFF="${KL_COEFF:-0.0}"

RESULTS_DIR="${RESULTS_DIR:-results/calibrate_v3}"
WORK_DIR="${WORK_DIR:-runs/calibrate_v3}"
mkdir -p "$RESULTS_DIR" "$WORK_DIR"

for DIFFICULTY in $DIFFICULTIES; do
    echo "############## calibrating difficulty=$DIFFICULTY ##############"
    DIFFICULTY="$DIFFICULTY" \
    ENV="$ENV" SPEND_MODE="$SPEND_MODE" \
    SEEDS="$SEEDS" BUDGETS="$BUDGETS" \
    NUM_EPISODES="$NUM_EPISODES" NUM_STEPS="$NUM_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" NUM_EVAL_EPISODES="$NUM_EVAL_EPISODES" \
    THRESHOLD="$THRESHOLD" KL_COEFF="$KL_COEFF" \
    RESULTS_DIR="$RESULTS_DIR/$DIFFICULTY" \
    WORK_DIR="$WORK_DIR/$DIFFICULTY" \
    bash "$APP_DIR/experiments/exp_budget_sweep.sh"
done

echo "############## combined calibration comparison ##############"
# Flat dump of every row across difficulties.
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/calibration_all.csv" \
    --markdown "$RESULTS_DIR/calibration_all.md"

# Comparison table: mean/std success + interventions + utilisation per
# (difficulty, budget). This is the table to read when picking a preset.
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --group-by difficulty,budget,spend_mode,env \
    --group-metrics success_rate,mean_interventions_used,budget_utilization,cost_per_uplift_point \
    --csv "$RESULTS_DIR/calibration_compare.csv" \
    --markdown "$RESULTS_DIR/calibration_compare.md"

echo "[exp_calibrate_v3] wrote comparison table to $RESULTS_DIR/calibration_compare.md"
echo "[exp_calibrate_v3] target curve: B0<=5%, B1 20-40%, B2 35-60%, B4 55-75%, B8 70-90%"
