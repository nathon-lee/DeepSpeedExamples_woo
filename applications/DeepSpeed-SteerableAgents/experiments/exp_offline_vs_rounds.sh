#!/usr/bin/env bash
# Experiment 3: offline vs rounds (online) training, matched compute.
#
# Multi-seed by default. Uses harder V3 + adaptive spend unless overridden.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"

ENV="${ENV:-v3}"
SPEND_MODE="${SPEND_MODE:-adaptive}"
DIFFICULTY="${DIFFICULTY:-medium}"
NUM_ACTIONS="${NUM_ACTIONS:-4}"
NUM_CRITICAL_NODES="${NUM_CRITICAL_NODES:-}"
STOCHASTICITY="${STOCHASTICITY:-}"
TRANSITION_NOISE="${TRANSITION_NOISE:-}"
INTERVENTION_EFFECT_SPAN="${INTERVENTION_EFFECT_SPAN:-}"
FAILURE_SOFTNESS="${FAILURE_SOFTNESS:-}"
REQUIRED_CRITICAL_PASSES="${REQUIRED_CRITICAL_PASSES:-}"
RISK_THRESHOLD="${RISK_THRESHOLD:-}"
MIN_GAP_BETWEEN_INTERVENTIONS="${MIN_GAP_BETWEEN_INTERVENTIONS:-}"

if [ -n "$DIFFICULTY" ] && [ "$ENV" = "v3" ]; then
    _preset_h="$(PYTHONPATH="$APP_DIR" python -c "from envs.toy_long_horizon_env_v3 import DIFFICULTY_PRESETS as P; print(P['$DIFFICULTY']['horizon'])")"
    HORIZON="${HORIZON:-$_preset_h}"
    EPISODE_STEPS="${EPISODE_STEPS:-$_preset_h}"
else
    HORIZON="${HORIZON:-40}"
    EPISODE_STEPS="${EPISODE_STEPS:-40}"
fi

NUM_EPISODES="${NUM_EPISODES:-512}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-256}"
THRESHOLD="${THRESHOLD:-0.95}"
KL_COEFF="${KL_COEFF:-0.0}"

SEEDS="${SEEDS:-0 1 2 3 4}"
B="${B:-4}"

ROUNDS="${ROUNDS:-10}"
NUM_STEPS_PER_ROUND="${NUM_STEPS_PER_ROUND:-200}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-64}"
GLOBAL_BUDGET_PER_ROUND="${GLOBAL_BUDGET_PER_ROUND:-$((B * EPISODES_PER_ROUND))}"

RESULTS_DIR="${RESULTS_DIR:-results/offline_vs_rounds}"
WORK_DIR="${WORK_DIR:-runs/offline_vs_rounds}"

mkdir -p "$RESULTS_DIR" "$WORK_DIR"
cd "$WORK_DIR"

for SEED in $SEEDS; do
    echo "=== offline_vs_rounds seed=$SEED spend_mode=$SPEND_MODE env=$ENV ==="

    # ----------------------------- offline arm -----------------------------
    t_start=$(date +%s)
    collect_offline=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES
        GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B
        THRESHOLD=$THRESHOLD SEED=$SEED SPEND_MODE=$SPEND_MODE
        DIFFICULTY=$DIFFICULTY
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS
        RISK_THRESHOLD=$RISK_THRESHOLD
        MIN_GAP_BETWEEN_INTERVENTIONS=$MIN_GAP_BETWEEN_INTERVENTIONS
        OUTPUT=roll_offline_s${SEED}.jsonl
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        collect_offline+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${collect_offline[@]}" bash "$APP_DIR/scripts/run_collect.sh"

    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE \
        SEED=$SEED KL_COEFF=$KL_COEFF SPEND_MODE=$SPEND_MODE \
        DIFFICULTY=$DIFFICULTY \
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES STOCHASTICITY=$STOCHASTICITY \
        TRANSITION_NOISE=$TRANSITION_NOISE \
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN \
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS \
        RISK_THRESHOLD=$RISK_THRESHOLD \
        MIN_GAP_BETWEEN_INTERVENTIONS=$MIN_GAP_BETWEEN_INTERVENTIONS \
        ${REQUIRED_CRITICAL_PASSES:+REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES} \
        ROLLOUTS=roll_offline_s${SEED}.jsonl \
        CHECKPOINT=ckpt_offline_s${SEED}.pt \
        bash "$APP_DIR/scripts/run_train.sh"

    eval_offline=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES
        DIFFICULTY=$DIFFICULTY
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS
        EVAL_SEED=$((SEED + 1000))
        ROLLOUTS=roll_offline_s${SEED}.jsonl
        CHECKPOINT=ckpt_offline_s${SEED}.pt
        EVAL_PREFIX=offline_s${SEED}_
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        eval_offline+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${eval_offline[@]}" bash "$APP_DIR/scripts/run_eval.sh"
    t_offline=$(( $(date +%s) - t_start ))

    python "$APP_DIR/experiments/result_schema.py" \
        --experiment offline_vs_rounds \
        --run-id "offline_s${SEED}" \
        --seed "$SEED" \
        --mode offline \
        --budget "$B" \
        --kl-coeff "$KL_COEFF" \
        --spend-mode "$SPEND_MODE" \
        --env "$ENV" \
        --difficulty "$DIFFICULTY" \
        --horizon "$HORIZON" \
        --episode-steps "$EPISODE_STEPS" \
        --num-actions "$NUM_ACTIONS" \
        --checkpoint "$(pwd)/ckpt_offline_s${SEED}.pt" \
        --eval-success "$(pwd)/offline_s${SEED}_eval_success.json" \
        --eval-budget "$(pwd)/offline_s${SEED}_eval_budget.json" \
        --eval-steerability "$(pwd)/offline_s${SEED}_eval_steerability.json" \
        --wall-time-sec "$t_offline" \
        --extra-json "{\"total_train_steps\":$NUM_STEPS,\"difficulty\":\"$DIFFICULTY\"}" \
        --out "../../$RESULTS_DIR/offline_s${SEED}.json"

    # ----------------------------- rounds arm ------------------------------
    t_start=$(date +%s)
    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_ACTIONS=$NUM_ACTIONS SPEND_MODE=$SPEND_MODE \
        DIFFICULTY=$DIFFICULTY \
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES STOCHASTICITY=$STOCHASTICITY \
        TRANSITION_NOISE=$TRANSITION_NOISE \
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN \
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS \
        RISK_THRESHOLD=$RISK_THRESHOLD \
        MIN_GAP_BETWEEN_INTERVENTIONS=$MIN_GAP_BETWEEN_INTERVENTIONS \
        ${REQUIRED_CRITICAL_PASSES:+REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES} \
        ROUNDS=$ROUNDS EPISODES_PER_ROUND=$EPISODES_PER_ROUND \
        NUM_STEPS_PER_ROUND=$NUM_STEPS_PER_ROUND \
        GLOBAL_BUDGET_PER_ROUND=$GLOBAL_BUDGET_PER_ROUND \
        PER_EP_BUDGET=$B THRESHOLD=$THRESHOLD \
        BATCH_SIZE=$BATCH_SIZE SEED=$SEED KL_COEFF=$KL_COEFF \
        CHECKPOINT=ckpt_rounds_s${SEED}.pt \
        bash "$APP_DIR/scripts/run_train.sh"

    collect_rounds_eval=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EPISODES=$NUM_EPISODES
        GLOBAL_BUDGET=$((B * NUM_EPISODES)) PER_EP_BUDGET=$B
        THRESHOLD=$THRESHOLD SEED=$((SEED + 2000)) SPEND_MODE=$SPEND_MODE
        DIFFICULTY=$DIFFICULTY
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS
        RISK_THRESHOLD=$RISK_THRESHOLD
        MIN_GAP_BETWEEN_INTERVENTIONS=$MIN_GAP_BETWEEN_INTERVENTIONS
        CHECKPOINT=ckpt_rounds_s${SEED}.pt
        OUTPUT=roll_rounds_eval_s${SEED}.jsonl
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        collect_rounds_eval+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${collect_rounds_eval[@]}" bash "$APP_DIR/scripts/run_collect.sh"

    eval_rounds=(
        ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS
        NUM_ACTIONS=$NUM_ACTIONS NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES
        DIFFICULTY=$DIFFICULTY
        NUM_CRITICAL_NODES=$NUM_CRITICAL_NODES
        STOCHASTICITY=$STOCHASTICITY
        TRANSITION_NOISE=$TRANSITION_NOISE
        INTERVENTION_EFFECT_SPAN=$INTERVENTION_EFFECT_SPAN
        FAILURE_SOFTNESS=$FAILURE_SOFTNESS
        EVAL_SEED=$((SEED + 1000))
        ROLLOUTS=roll_rounds_eval_s${SEED}.jsonl
        CHECKPOINT=ckpt_rounds_s${SEED}.pt
        EVAL_PREFIX=rounds_s${SEED}_
    )
    if [ -n "$REQUIRED_CRITICAL_PASSES" ]; then
        eval_rounds+=(REQUIRED_CRITICAL_PASSES=$REQUIRED_CRITICAL_PASSES)
    fi
    env "${eval_rounds[@]}" bash "$APP_DIR/scripts/run_eval.sh"
    t_rounds=$(( $(date +%s) - t_start ))

    python "$APP_DIR/experiments/result_schema.py" \
        --experiment offline_vs_rounds \
        --run-id "rounds_s${SEED}" \
        --seed "$SEED" \
        --mode rounds \
        --budget "$B" \
        --kl-coeff "$KL_COEFF" \
        --spend-mode "$SPEND_MODE" \
        --env "$ENV" \
        --difficulty "$DIFFICULTY" \
        --horizon "$HORIZON" \
        --episode-steps "$EPISODE_STEPS" \
        --num-actions "$NUM_ACTIONS" \
        --checkpoint "$(pwd)/ckpt_rounds_s${SEED}.pt" \
        --eval-success "$(pwd)/rounds_s${SEED}_eval_success.json" \
        --eval-budget "$(pwd)/rounds_s${SEED}_eval_budget.json" \
        --eval-steerability "$(pwd)/rounds_s${SEED}_eval_steerability.json" \
        --wall-time-sec "$t_rounds" \
        --extra-json "{\"total_train_steps\":$((ROUNDS*NUM_STEPS_PER_ROUND)),\"rounds\":$ROUNDS,\"episodes_per_round\":$EPISODES_PER_ROUND,\"difficulty\":\"$DIFFICULTY\"}" \
        --out "../../$RESULTS_DIR/rounds_s${SEED}.json"
done

cd - > /dev/null
python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/summary.csv" \
    --markdown "$RESULTS_DIR/summary.md"

python "$APP_DIR/experiments/aggregate_results.py" "$RESULTS_DIR" \
    --group-by experiment,mode,budget,kl_coeff,spend_mode,env,difficulty \
    --csv "$RESULTS_DIR/summary_seed_stats.csv" \
    --markdown "$RESULTS_DIR/summary_seed_stats.md"
