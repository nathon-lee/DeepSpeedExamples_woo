#!/usr/bin/env bash
# Experiment 2: BC-only vs BC+KL distillation ablation.
#
# Holds (env, B, seed, num_steps) fixed, varies only --kl-coeff in {0, 0.5}.
# Same rollouts file is reused so the only difference between the two
# checkpoints is the loss function.
#
# Writes:
#   results/distill_ablation/bc_s{SEED}.json     (kl_coeff=0)
#   results/distill_ablation/bc_kl_s{SEED}.json  (kl_coeff=0.5)
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
B="${B:-4}"
THRESHOLD="${THRESHOLD:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/distill_ablation}"
WORK_DIR="${WORK_DIR:-runs/distill_ablation}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

# Single shared collection for both ablation arms.
ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES \
    GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B \
    THRESHOLD=$THRESHOLD SEED=$SEED \
    OUTPUT=roll_ablation.jsonl \
    bash "$APP_DIR/scripts/run_collect.sh"

run_one() {
    local tag="$1"            # "bc" or "bc_kl"
    local kl="$2"             # 0.0 or 0.5
    local ckpt="ckpt_${tag}.pt"

    local t_start; t_start=$(date +%s)

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
        SEED=$SEED KL_COEFF="$kl" \
        ROLLOUTS=roll_ablation.jsonl \
        CHECKPOINT="$ckpt" \
        bash "$APP_DIR/scripts/run_train.sh"

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES \
        EVAL_SEED=$((SEED + 1000)) \
        ROLLOUTS=roll_ablation.jsonl \
        CHECKPOINT="$ckpt" \
        EVAL_PREFIX="${tag}_" \
        bash "$APP_DIR/scripts/run_eval.sh"

    local t_end; t_end=$(date +%s)

    python "$APP_DIR/experiments/result_schema.py" \
        --experiment distill_ablation \
        --run-id "${tag}_s${SEED}" \
        --seed "$SEED" \
        --mode offline \
        --budget "$B" \
        --kl-coeff "$kl" \
        --env "$ENV" \
        --horizon "$HORIZON" \
        --episode-steps "$EPISODE_STEPS" \
        --num-actions "$NUM_ACTIONS" \
        --checkpoint "$(pwd)/$ckpt" \
        --eval-success "$(pwd)/${tag}_eval_success.json" \
        --eval-budget "$(pwd)/${tag}_eval_budget.json" \
        --eval-steerability "$(pwd)/${tag}_eval_steerability.json" \
        --wall-time-sec "$((t_end - t_start))" \
        --extra-json "{\"variant\":\"$tag\"}" \
        --out "../../$RESULTS_DIR/${tag}_s${SEED}.json"
}

run_one bc    0.0
run_one bc_kl 0.5

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"
