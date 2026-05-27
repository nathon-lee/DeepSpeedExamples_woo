#!/usr/bin/env bash
# Run all three eval scripts and emit JSON reports.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"
cd "$APP_DIR"
ROLLOUTS="${ROLLOUTS:-rollouts.jsonl}"
CKPT="${CHECKPOINT:-checkpoints/student.pt}"

python eval/eval_success.py \
    --checkpoint "$CKPT" \
    --num-episodes "${NUM_EVAL_EPISODES:-128}" \
    --horizon "${HORIZON:-32}" \
    --num-actions "${NUM_ACTIONS:-4}" \
    --seed "${EVAL_SEED:-1234}" \
    --env "${ENV:-v1}" \
    --output eval_success.json

python eval/eval_budget.py --rollouts "$ROLLOUTS" --output eval_budget.json
python eval/eval_steerability.py --rollouts "$ROLLOUTS" --output eval_steerability.json
