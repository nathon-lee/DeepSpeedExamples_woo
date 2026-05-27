#!/usr/bin/env bash
# Experiment 1: per-episode budget sweep on V2.
#
# For each B in BUDGETS:
#   * collect with PER_EP_BUDGET=B
#   * train an offline student (per-budget checkpoint!)
#   * eval
#   * write one result row to results/budget_sweep/B{B}_s{SEED}.json
#
# Then aggregate into a CSV / Markdown summary.
#
# All paths resolve against the *caller's* cwd. Configure via env vars:
#   ENV (default v2), HORIZON, EPISODE_STEPS, NUM_ACTIONS, NUM_EPISODES,
#   NUM_STEPS, BATCH_SIZE, NUM_EVAL_EPISODES, SEED, BUDGETS, RESULTS_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"

ENV="${ENV:-v2}"
HORIZON="${HORIZON:-8}"
EPISODE_STEPS="${EPISODE_STEPS:-24}"
NUM_ACTIONS="${NUM_ACTIONS:-4}"
NUM_EPISODES="${NUM_EPISODES:-512}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-256}"
SEED="${SEED:-0}"
THRESHOLD="${THRESHOLD:-0}"
BUDGETS="${BUDGETS:-0 1 2 4 8}"
RESULTS_DIR="${RESULTS_DIR:-results/budget_sweep}"
WORK_DIR="${WORK_DIR:-runs/budget_sweep}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

baseline_succ=""
for B in $BUDGETS; do
    echo "=== budget=$B seed=$SEED ==="
    t_start=$(date +%s)

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES \
        GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B \
        THRESHOLD=$THRESHOLD SEED=$SEED \
        OUTPUT=roll_b${B}.jsonl \
        bash "$APP_DIR/scripts/run_collect.sh"

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
        SEED=$SEED \
        ROLLOUTS=roll_b${B}.jsonl \
        CHECKPOINT=ckpt_b${B}.pt \
        bash "$APP_DIR/scripts/run_train.sh"

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES \
        EVAL_SEED=$((SEED + 1000)) \
        ROLLOUTS=roll_b${B}.jsonl \
        CHECKPOINT=ckpt_b${B}.pt \
        EVAL_PREFIX=b${B}_ \
        bash "$APP_DIR/scripts/run_eval.sh"

    t_end=$(date +%s)

    # First run (B=0) is the baseline; subsequent rows reference it.
    extra_args=()
    if [ -n "$baseline_succ" ]; then
        extra_args+=(--baseline-success-rate "$baseline_succ")
    fi
    python "$APP_DIR/experiments/result_schema.py" \
        --experiment budget_sweep \
        --run-id "B${B}_s${SEED}" \
        --seed "$SEED" \
        --mode offline \
        --budget "$B" \
        --kl-coeff "${KL_COEFF:-0.5}" \
        --env "$ENV" \
        --horizon "$HORIZON" \
        --episode-steps "$EPISODE_STEPS" \
        --num-actions "$NUM_ACTIONS" \
        --checkpoint "$(pwd)/ckpt_b${B}.pt" \
        --eval-success "$(pwd)/b${B}_eval_success.json" \
        --eval-budget "$(pwd)/b${B}_eval_budget.json" \
        --eval-steerability "$(pwd)/b${B}_eval_steerability.json" \
        --wall-time-sec "$((t_end - t_start))" \
        "${extra_args[@]}" \
        --out "../../$RESULTS_DIR/B${B}_s${SEED}.json"

    if [ -z "$baseline_succ" ]; then
        baseline_succ=$(python -c \
            "import json; print(json.load(open('b${B}_eval_success.json'))['success_rate'])")
        echo "[exp_budget_sweep] baseline success_rate=$baseline_succ"
    fi
done

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"
