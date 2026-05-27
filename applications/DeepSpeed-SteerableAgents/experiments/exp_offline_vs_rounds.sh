#!/usr/bin/env bash
# Experiment 3: offline vs rounds (online) training, matched compute.
#
# Both arms use:
#   * the same env config
#   * the same total optimisation budget (NUM_STEPS == ROUNDS*NUM_STEPS_PER_ROUND)
#   * the same per-episode budget at COLLECT time
#
# Offline arm is the standard collect -> train -> eval pipeline.
# Rounds arm uses the in-process collect/train alternation; for a fair eval
# we re-run a held-out collection at the end (so eval_budget /
# eval_steerability see comparable trajectories).
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

# Rounds-mode knobs (must satisfy ROUNDS*NUM_STEPS_PER_ROUND == NUM_STEPS).
ROUNDS="${ROUNDS:-10}"
NUM_STEPS_PER_ROUND="${NUM_STEPS_PER_ROUND:-200}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-64}"
GLOBAL_BUDGET_PER_ROUND="${GLOBAL_BUDGET_PER_ROUND:-$((B * EPISODES_PER_ROUND))}"

RESULTS_DIR="${RESULTS_DIR:-results/offline_vs_rounds}"
WORK_DIR="${WORK_DIR:-runs/offline_vs_rounds}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

# ----------------------------- offline arm -------------------------------
t_start=$(date +%s)
ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES \
    GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B \
    THRESHOLD=$THRESHOLD SEED=$SEED \
    OUTPUT=roll_offline.jsonl \
    bash "$APP_DIR/scripts/run_collect.sh"

ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
    SEED=$SEED \
    ROLLOUTS=roll_offline.jsonl \
    CHECKPOINT=ckpt_offline.pt \
    bash "$APP_DIR/scripts/run_train.sh"

ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES \
    EVAL_SEED=$((SEED + 1000)) \
    ROLLOUTS=roll_offline.jsonl \
    CHECKPOINT=ckpt_offline.pt \
    EVAL_PREFIX=offline_ \
    bash "$APP_DIR/scripts/run_eval.sh"
t_offline=$(( $(date +%s) - t_start ))

python "$APP_DIR/experiments/result_schema.py" \
    --experiment offline_vs_rounds \
    --run-id "offline_s${SEED}" \
    --seed "$SEED" \
    --mode offline \
    --budget "$B" \
    --kl-coeff "${KL_COEFF:-0.5}" \
    --env "$ENV" \
    --horizon "$HORIZON" \
    --episode-steps "$EPISODE_STEPS" \
    --num-actions "$NUM_ACTIONS" \
    --checkpoint "$(pwd)/ckpt_offline.pt" \
    --eval-success "$(pwd)/offline_eval_success.json" \
    --eval-budget "$(pwd)/offline_eval_budget.json" \
    --eval-steerability "$(pwd)/offline_eval_steerability.json" \
    --wall-time-sec "$t_offline" \
    --extra-json "{\"total_train_steps\":$NUM_STEPS}" \
    --out "../../$RESULTS_DIR/offline_s${SEED}.json"

# ----------------------------- rounds arm --------------------------------
t_start=$(date +%s)
ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS \
    ROUNDS=$ROUNDS EPISODES_PER_ROUND=$EPISODES_PER_ROUND \
    NUM_STEPS_PER_ROUND=$NUM_STEPS_PER_ROUND \
    GLOBAL_BUDGET_PER_ROUND=$GLOBAL_BUDGET_PER_ROUND \
    PER_EP_BUDGET=$B THRESHOLD=$THRESHOLD \
    BATCH_SIZE=$BATCH_SIZE SEED=$SEED \
    CHECKPOINT=ckpt_rounds.pt \
    bash "$APP_DIR/scripts/run_train.sh"

# Held-out evaluation collection with the trained rounds student so the
# eval_budget / eval_steerability rollouts file is matched to the
# now-finished checkpoint (not biased by the random-init round 1).
ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES \
    GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B \
    THRESHOLD=$THRESHOLD SEED=$((SEED + 2000)) \
    CHECKPOINT=ckpt_rounds.pt \
    OUTPUT=roll_rounds_eval.jsonl \
    bash "$APP_DIR/scripts/run_collect.sh"

ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
    NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES \
    EVAL_SEED=$((SEED + 1000)) \
    ROLLOUTS=roll_rounds_eval.jsonl \
    CHECKPOINT=ckpt_rounds.pt \
    EVAL_PREFIX=rounds_ \
    bash "$APP_DIR/scripts/run_eval.sh"
t_rounds=$(( $(date +%s) - t_start ))

python "$APP_DIR/experiments/result_schema.py" \
    --experiment offline_vs_rounds \
    --run-id "rounds_s${SEED}" \
    --seed "$SEED" \
    --mode rounds \
    --budget "$B" \
    --kl-coeff "${KL_COEFF:-0.5}" \
    --env "$ENV" \
    --horizon "$HORIZON" \
    --episode-steps "$EPISODE_STEPS" \
    --num-actions "$NUM_ACTIONS" \
    --checkpoint "$(pwd)/ckpt_rounds.pt" \
    --eval-success "$(pwd)/rounds_eval_success.json" \
    --eval-budget "$(pwd)/rounds_eval_budget.json" \
    --eval-steerability "$(pwd)/rounds_eval_steerability.json" \
    --wall-time-sec "$t_rounds" \
    --extra-json "{\"total_train_steps\":$((ROUNDS*NUM_STEPS_PER_ROUND)),\"rounds\":$ROUNDS,\"episodes_per_round\":$EPISODES_PER_ROUND}" \
    --out "../../$RESULTS_DIR/rounds_s${SEED}.json"

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"
