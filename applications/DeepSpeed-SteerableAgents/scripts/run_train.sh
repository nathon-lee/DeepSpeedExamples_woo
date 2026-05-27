#!/usr/bin/env bash
# Train the student via online policy distillation on the replay buffer.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"
cd "$APP_DIR"
python training/online_distill_trainer.py \
    --rollouts "${ROLLOUTS:-rollouts.jsonl}" \
    --ds-config "${DS_CONFIG:-configs/distill_ds_config.json}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --num-steps "${NUM_STEPS:-500}" \
    --capacity "${CAPACITY:-100000}" \
    --seed "${SEED:-0}" \
    --output-dir "${OUTPUT_DIR:-checkpoints}" \
    --horizon "${HORIZON:-32}" \
    --num-actions "${NUM_ACTIONS:-4}" \
    --env "${ENV:-v1}" \
    ${PRIORITIZED:+--prioritized}
