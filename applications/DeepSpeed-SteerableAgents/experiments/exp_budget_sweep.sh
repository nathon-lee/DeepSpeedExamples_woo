#!/usr/bin/env bash
# Experiment 1: per-episode budget sweep (multi-seed).
#
# Defaults target the harder V3 environment with adaptive spend mode.
# Override via env vars as needed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"

ENV="${ENV:-v3}"
SPEND_MODE="${SPEND_MODE:-adaptive}"
HORIZON="${HORIZON:-40}"
EPISODE_STEPS="${EPISODE_STEPS:-40}"
NUM_ACTIONS="${NUM_ACTIONS:-4}"
NUM_CRITICAL_NODES="${NUM_CRITICAL_NODES:-4}"
STOCHASTICITY="${STOCHASTICITY:-0.25}"
TRANSITION_NOISE="${TRANSITION_NOISE:-0.10}"
REQUIRED_CRITICAL_PASSES="${REQUIRED_CRITICAL_PASSES:-}"

NUM_EPISODES="${NUM_EPISODES:-512}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-256}"
THRESHOLD="${THRESHOLD:-0.95}"
KL_COEFF="${KL_COEFF:-0.0}"

SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGETS="${BUDGETS:-0 1 2 4 8}"
RESULTS_DIR="${RESULTS_DIR:-results/budget_sweep}"
WORK_DIR="${WORK_DIR:-runs/budget_sweep}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

for SEED in $SEEDS; do
    baseline_succ=""
    for B in $BUDGETS; do
        echo "=== budget=$B seed=$SEED spend_mode=$SPEND_MODE env=$ENV ==="
        t_start=$(date +%s)

        collect_cmd=(
            ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
            NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES
            GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B
            THRESHOLD=$THRESHOLD SEED=$SEED SPEND_MODE=$SPEND_MODE
            NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
            STOCHASTICITY=$STOCHASTICITY
            TRANSITION_NOISE=$TRANSITION_NOISE
            OUTPUT=roll_b${B}_s${SEED}.jsonl
        )
        if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
            collect_cmd+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
        fi
        env "${collect_cmd[@]}" bash "$APP_DIR/scripts/run_collect.sh"

        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
            NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
            SEED=$SEED KL_COEFF=$KL_COEFF SPEND_MODE=$SPEND_MODE \
            NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES \
            STOCHASTICITY=$STOCHASTICITY TRANSITION_NOISE=$TRANSITION_NOISE \
            ROLLOUTS=roll_b${B}_s${SEED}.jsonl \
            CHECKPOINT=ckpt_b${B}_s${SEED}.pt \
            bash "$APP_DIR/scripts/run_train.sh"

        eval_cmd=(
            ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
            NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES
            NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
            STOCHASTICITY=$STOCHASTICITY
            TRANSITION_NOISE=$TRANSITION_NOISE
            EVAL_SEED=$((SEED + 1000))
            ROLLOUTS=roll_b${B}_s${SEED}.jsonl
            CHECKPOINT=ckpt_b${B}_s${SEED}.pt
            EVAL_PREFIX=b${B}_s${SEED}_
        )
        if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
            eval_cmd+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
        fi
        env "${eval_cmd[@]}" bash "$APP_DIR/scripts/run_eval.sh"

        t_end=$(date +%s)

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
            --kl-coeff "$KL_COEFF" \
            --spend-mode "$SPEND_MODE" \
            --env "$ENV" \
            --horizon "$HORIZON" \
            --episode-steps "$EPISODE_STEPS" \
            --num-actions "$NUM_ACTIONS" \
            --checkpoint "$(pwd)/ckpt_b${B}_s${SEED}.pt" \
            --eval-success "$(pwd)/b${B}_s${SEED}_eval_success.json" \
            --eval-budget "$(pwd)/b${B}_s${SEED}_eval_budget.json" \
            --eval-steerability "$(pwd)/b${B}_s${SEED}_eval_steerability.json" \
            --wall-time-sec "$((t_end - t_start))" \
            "${extra_args[@]}" \
            --extra-json "{\"num_critical_nodes\":$NUM_CRITICAL_NODES,\"stochasticity\":$STOCHASTICITY,\"transition_noise\":$TRANSITION_NOISE}" \
            --out "../../$RESULTS_DIR/B${B}_s${SEED}.json"

        if [ -z "$baseline_succ" ]; then
            baseline_succ=$(python -c "import json; print(json.load(open('b${B}_s${SEED}_eval_success.json'))['success_rate'])")
            echo "[exp_budget_sweep] seed=$SEED baseline success_rate=$baseline_succ"
        fi
    done

done

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"

python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --group-by experiment,mode,budget,kl_coeff,spend_mode,env \
    --csv "$RESULTS_DIR/summary_seed_stats.csv" \
    --markdown "$RESULTS_DIR/summary_seed_stats.md"
