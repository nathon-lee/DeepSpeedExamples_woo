#!/usr/bin/env bash
# Collect steered rollouts using the toy long-horizon env.
#
# All input/output paths are interpreted relative to the *caller's* working
# directory (we do NOT cd into APP_DIR), so this script composes cleanly with
# a sweep loop run from a sub-directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"
python "$APP_DIR/training/collect_rollouts.py" \
    --num-episodes "${NUM_EPISODES:-64}" \
    --horizon "${HORIZON:-32}" \
    --num-actions "${NUM_ACTIONS:-4}" \
    --global-budget "${GLOBAL_BUDGET:-128}" \
    --per-episode-budget "${PER_EP_BUDGET:-4}" \
    --threshold "${THRESHOLD:-0.6}" \
    --seed "${SEED:-0}" \
    --env "${ENV:-v1}" \
    ${EPISODE_STEPS:+--episode-steps "$EPISODE_STEPS"} \
    --spend-mode "${SPEND_MODE:-adaptive}" \
    ${DIFFICULTY:+--difficulty "$DIFFICULTY"} \
    ${NUM_CRITICAL_NODES:+--num-critical-nodes "$NUM_CRITICAL_NODES"} \
    ${STOCHASTICITY:+--stochasticity "$STOCHASTICITY"} \
    ${TRANSITION_NOISE:+--transition-noise "$TRANSITION_NOISE"} \
    ${INTERVENTION_EFFECT_SPAN:+--intervention-effect-span "$INTERVENTION_EFFECT_SPAN"} \
    ${FAILURE_SOFTNESS:+--failure-softness "$FAILURE_SOFTNESS"} \
    ${REQUIRED_CRITICAL_PASSES:+--required-critical-passes "$REQUIRED_CRITICAL_PASSES"} \
    ${RISK_THRESHOLD:+--risk-threshold "$RISK_THRESHOLD"} \
    ${MIN_GAP_BETWEEN_INTERVENTIONS:+--min-gap-between-interventions "$MIN_GAP_BETWEEN_INTERVENTIONS"} \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --output "${OUTPUT:-rollouts.jsonl}"
