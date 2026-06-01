# Experiment Report — calibrated setting (v3 + adaptive spend + difficulty presets)

Scope of this round:
- Calibrate v3 to a mid-difficulty regime so budget has discriminative power.
- Make intervention budget a true cap (adaptive spend).
- Report all core experiments over 5 seeds with mean ± std, tagged by difficulty.

## Difficulty presets

`envs/toy_long_horizon_env_v3.py` ships three presets (explicit knobs override):

| preset | nodes | horizon | stochasticity | transition_noise | span | softness | required passes |
|---|---|---|---|---|---|---|---|
| easy   | 2 | 16 | 0.05 | 0.05 | 2 | high   | ceil(0.5·n)=1/2 |
| medium | 3 | 24 | 0.10 | 0.08 | 1 | medium | ceil(0.6·n)=2/3 |
| hard   | 5 | 32 | 0.15 | 0.10 | 1 | low    | ceil(0.8·n)=4/5 |

Key calibration levers (relative to the prior too-hard v3):
- **Soft failure**: success needs only a fraction of critical nodes, not all of
  them (`failure_softness` / `required_critical_passes`).
- **Reliable, local interventions**: a steered step suppresses transition noise
  and `intervention_effect_span` grants a short local follow-up advantage — but
  never solves all future nodes.
- **Smarter adaptive spend**: `risk_threshold` / `min_gap_between_interventions`
  steer the limited budget toward critical / risky states.

## Config used in this round

Environment:
- env = `v3`
- difficulty = `medium` (paper-facing default; sweep all three to pick)

Intervention control:
- spend_mode = `adaptive` (default for this round)
- forced mode still available via `SPEND_MODE=forced`
- optional `RISK_THRESHOLD`, `MIN_GAP_BETWEEN_INTERVENTIONS`

Training/eval:
- num_steps = 2000
- batch_size = 128
- num_eval_episodes = 256
- seeds = {0,1,2,3,4}

## Repro commands

First calibrate (seed 0, all three presets) and read the comparison table:

```bash
# Calibration: env=v3, adaptive, budgets {0,1,2,4,8}, presets easy/medium/hard
bash experiments/exp_calibrate_v3.sh
# -> results/calibrate_v3/calibration_compare.md
```

Then run the three experiment groups at the chosen difficulty (default medium):

```bash
# 1) budget sweep (5 seeds)
SEEDS="0 1 2 3 4" ENV=v3 SPEND_MODE=adaptive DIFFICULTY=medium \
BUDGETS="0 1 2 4 8" \
bash experiments/exp_budget_sweep.sh

# 2) distillation ablation (5 seeds)
SEEDS="0 1 2 3 4" ENV=v3 SPEND_MODE=adaptive DIFFICULTY=medium \
B=4 KL_BC=0.0 KL_BC_KL=0.5 \
bash experiments/exp_distill_ablation.sh

# 3) offline vs rounds (5 seeds)
SEEDS="0 1 2 3 4" ENV=v3 SPEND_MODE=adaptive DIFFICULTY=medium \
B=4 KL_COEFF=0.0 \
ROUNDS=10 NUM_STEPS_PER_ROUND=200 EPISODES_PER_ROUND=64 \
bash experiments/exp_offline_vs_rounds.sh
```

Aggregate flat + seed stats:

```bash
python experiments/aggregate_results.py results/ \
  --csv results/all.csv --markdown results/all.md

python experiments/aggregate_results.py results/ \
  --group-by experiment,mode,budget,kl_coeff,spend_mode,env,difficulty \
  --csv results/all_seed_stats.csv \
  --markdown results/all_seed_stats.md
```

Generate plots:

```bash
python experiments/plot_results.py \
  --csv results/all.csv --env v3 --spend-mode adaptive --out-dir plots/
```

## Budget sweep summary (multi-seed)

Paste from:
- `results/budget_sweep/summary_seed_stats.md`

Expected columns:
- success_rate_mean / success_rate_std / success_rate_n
- mean_interventions_used_mean / std / n
- budget_utilization_mean / std / n
- cost_per_uplift_point_mean / std / n

Interpretation checklist:
- Is success vs budget now gradual (not instantly saturated)?
- Is budget utilization < 1.0 at least for some budgets (adaptive spend working)?
- Is there monotonic saturation or an interior optimum?

## Distillation ablation summary (harder setting)

Paste from:
- `results/distill_ablation/summary_seed_stats.md`

Report:
- BC-only (kl=0.0): mean ± std
- BC+KL (kl=0.5): mean ± std
- Difference in pp and whether it is stable across seeds.
- Include untrained baseline reference from budget B=0 if needed.

## Offline vs rounds summary (harder setting)

Paste from:
- `results/offline_vs_rounds/summary_seed_stats.md`

Report:
- Offline: mean ± std
- Rounds: mean ± std
- Whether rounds beats or matches offline under matched compute.

## Ceiling / floor sanity checks

The aggregator emits:
- a **ceiling** warning if all rows have success_rate >= 0.98 (too easy), and
- a **floor** warning if all budgeted (B>=1) rows have success_rate <= 0.02
  (too hard / steering too weak).

If the ceiling warning appears, move to a harder preset / raise knobs
(`num_critical_nodes`, `stochasticity`, `transition_noise`,
`required_critical_passes`). If the floor warning appears, move to an easier
preset / lower `required_critical_passes`, raise `failure_softness`, or raise
`intervention_effect_span`.

## Calibration: picking a preset

Read `results/calibrate_v3/calibration_compare.md` (grouped by difficulty,
budget). Pick the preset whose success-vs-budget curve is closest to the
target:

| budget | target success |
|---|---|
| 0 | 0–5% |
| 1 | 20–40% |
| 2 | 35–60% |
| 4 | 55–75% |
| 8 | 70–90% |

## Explicit note on previous B=8 regression claim

The prior toy-v2 finding ("B=8 regresses vs B=2") was likely noise under a
ceilinged setup where nearly all configurations saturated. This round uses the
harder v3 + adaptive-spend protocol specifically to re-test that claim under a
non-saturated regime. Whether the regression reappears should be determined
from `results/budget_sweep/summary_seed_stats.md` after rerun.

## Output files

- `results/all.csv`
- `results/all.md`
- `results/all_seed_stats.csv`
- `results/all_seed_stats.md`
- `results/calibrate_v3/calibration_all.{csv,md}`
- `results/calibrate_v3/calibration_compare.{csv,md}`
- `results/budget_sweep/summary_seed_stats.{csv,md}`
- `results/distill_ablation/summary_seed_stats.{csv,md}`
- `results/offline_vs_rounds/summary_seed_stats.{csv,md}`
- `plots/success_vs_budget.png`
- `plots/cost_per_uplift.png`
- `plots/distill_ablation.png`
- `plots/offline_vs_rounds.png`
