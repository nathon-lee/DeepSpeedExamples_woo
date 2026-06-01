#!/usr/bin/env bash
# Experiment 2: BC-only vs BC+KL distillation ablation (multi-seed).
#
# Defaults target harder V3 + adaptive spend. The two arms share one
# collection per seed; only KL coefficient changes.
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
B="${B:-4}"
SEEDS="${SEEDS:-0 1 2 3 4}"

KL_BC="${KL_BC:-0.0}"
KL_BC_KL="${KL_BC_KL:-0.5}"

RESULTS_DIR="${RESULTS_DIR:-results/distill_ablation}"
WORK_DIR="${WORK_DIR:-runs/distill_ablation}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

run_one() {
    local seed="$1"
    local tag="$2"      # bc | bc_kl
    local kl="$3"
    local ckpt="ckpt_${tag}_s${seed}.pt"

    local t_start; t_start=$(date +%s)

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
        SEED=$seed KL_COEFF="$kl" SPEND_MODE=$SPEND_MODE \
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES STOCHASTICITY=$STOCHASTICITY \
        TRANSITION_NOISE=$TRANSITION_NOISE \
        ${REQUIRED_CRITICAL_PASSES:+REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES} \
        ROLLOUTS=roll_ablation_s${seed}.jsonl \
        CHECKPOINT="$ckpt" \
        bash "$APP_DIR/scripts/run_train.sh"

    eval_cmd=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        EVAL_SEED=$((seed + 1000))
        ROLLOUTS=roll_ablation_s${seed}.jsonl
        CHECKPOINT=$ckpt
        EVAL_PREFIX=${tag}_s${seed}_
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        eval_cmd+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${eval_cmd[@]}" bash "$APP_DIR/scripts/run_eval.sh"

    local t_end; t_end=$(date +%s)

    python "$APP_DIR/experiments/result_schema.py" \
        --experiment distill_ablation \
        --run-id "${tag}_s${seed}" \
        --seed "$seed" \
        --mode offline \
        --budget "$B" \
        --kl-coeff "$kl" \
        --spend-mode "$SPEND_MODE" \
        --env "$ENV" \
        --horizon "$HORIZON" \
        --episode-steps "$EPISODE_STEPS" \
        --num-actions "$NUM_ACTIONS" \
        --checkpoint "$(pwd)/$ckpt" \
        --eval-success "$(pwd)/${tag}_s${seed}_eval_success.json" \
        --eval-budget "$(pwd)/${tag}_s${seed}_eval_budget.json" \
        --eval-steerability "$(pwd)/${tag}_s${seed}_eval_steerability.json" \
        --wall-time-sec "$((t_end - t_start))" \
        --extra-json "{\"variant\":\"$tag\",\"num_critical_nodes\":$NUM_CRITICAL_NODES,\"stochasticity\":$STOCHASTICITY,\"transition_noise\":$TRANSITION_NOISE}" \
        --out "../../$RESULTS_DIR/${tag}_s${seed}.json"
}

for SEED in $SEEDS; do
    echo "=== distill ablation seed=$SEED spend_mode=$SPEND_MODE env=$ENV ==="

    collect_cmd=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES
        GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B
        THRESHOLD=$THRESHOLD SEED=$SEED SPEND_MODE=$SPEND_MODE
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        OUTPUT=roll_ablation_s${SEED}.jsonl
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        collect_cmd+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${collect_cmd[@]}" bash "$APP_DIR/scripts/run_collect.sh"

    run_one "$SEED" bc "$KL_BC"
    run_one "$SEED" bc_kl "$KL_BC_KL"
done

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"

python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --group-by experiment,mode,budget,kl_coeff,spend_mode,env \
    --csv "$RESULTS_DIR/summary_seed_stats.csv" \
    --markdown "$RESULTS_DIR/summary_seed_stats.md"
