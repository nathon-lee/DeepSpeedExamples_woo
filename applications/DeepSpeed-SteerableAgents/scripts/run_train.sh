#!/usr/bin/env bash
# Train the student via online policy distillation on the replay buffer.
#
# All input/output paths are interpreted relative to the *caller's* working
# directory. CHECKPOINT, when set, becomes the full output path of the saved
# checkpoint (use this for per-budget sweeps).
#
# Two modes:
#   * Offline (default): reads ROLLOUTS jsonl once, trains NUM_STEPS updates.
#   * Rounds  (ROUNDS>0): alternates collect -> extend buffer -> train
#                         NUM_STEPS_PER_ROUND, for ROUNDS rounds, using
#                         the *current* student to drive the collector.
#                         GLOBAL_BUDGET_PER_ROUND / PER_EP_BUDGET / THRESHOLD
#                         control the per-round collection budget.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$HERE")"
python "$APP_DIR/training/online_distill_trainer.py" \
    --rollouts "${ROLLOUTS:-rollouts.jsonl}" \
    --ds-config "${DS_CONFIG:-$APP_DIR/configs/distill_ds_config.json}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --num-steps "${NUM_STEPS:-500}" \
    --capacity "${CAPACITY:-100000}" \
    --seed "${SEED:-0}" \
    --output-dir "${OUTPUT_DIR:-checkpoints}" \
    ${CHECKPOINT:+--checkpoint-path "$CHECKPOINT"} \
    --horizon "${HORIZON:-32}" \
    --num-actions "${NUM_ACTIONS:-4}" \
    --env "${ENV:-v1}" \
    ${EPISODE_STEPS:+--episode-steps "$EPISODE_STEPS"} \
    --rounds "${ROUNDS:-0}" \
    --episodes-per-round "${EPISODES_PER_ROUND:-64}" \
    --num-steps-per-round "${NUM_STEPS_PER_ROUND:-200}" \
    --global-budget-per-round "${GLOBAL_BUDGET_PER_ROUND:-128}" \
    --per-episode-budget "${PER_EP_BUDGET:-4}" \
    --threshold "${THRESHOLD:-0.6}" \
    ${SEED_ROLLOUTS:+--seed-rollouts "$SEED_ROLLOUTS"} \
    ${PRIORITIZED:+--prioritized}
