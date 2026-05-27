#!/usr/bin/env bash
# Run all three eval scripts and emit JSON reports.
#
# All input/output paths are interpreted relative to the *caller's* working
# directory. Set EVAL_PREFIX to add a prefix to the three report filenames
# (useful when looping over budgets).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"
ROLLOUTS="${ROLLOUTS:-rollouts.jsonl}"
CKPT="${CHECKPOINT:-checkpoints/student.pt}"
PREFIX="${EVAL_PREFIX:-}"

python "$APP_DIR/eval/eval_success.py" \
    --checkpoint "$CKPT" \
    --num-episodes "${NUM_EVAL_EPISODES:-128}" \
    --horizon "${HORIZON:-32}" \
    --num-actions "${NUM_ACTIONS:-4}" \
    --seed "${EVAL_SEED:-1234}" \
    --env "${ENV:-v1}" \
    ${EPISODE_STEPS:+--episode-steps "$EPISODE_STEPS"} \
    ${ALLOW_RANDOM_INIT:+--allow-random-init} \
    --output "${PREFIX}eval_success.json"

python "$APP_DIR/eval/eval_budget.py" \
    --rollouts "$ROLLOUTS" --output "${PREFIX}eval_budget.json"
python "$APP_DIR/eval/eval_steerability.py" \
    --rollouts "$ROLLOUTS" --output "${PREFIX}eval_steerability.json"
