# Experiment Report — harder setting (v3 + adaptive spend)

Scope of this round:
- Break the ceiling effect by increasing task difficulty.
- Make intervention budget a true cap (adaptive spend).
- Report all core experiments over 5 seeds with mean ± std.

## Config used in this round

Environment:
- env = `v3`
- horizon = 40
- episode_steps = 40
- num_actions = 4
- num_critical_nodes = 4
- stochasticity = 0.25
- transition_noise = 0.10
- required_critical_passes = (unset -> defaults to num_critical_nodes)

Intervention control:
- spend_mode = `adaptive` (default for this round)
- forced mode still available via `SPEND_MODE=forced`

Training/eval:
- num_steps = 2000
- batch_size = 128
- num_eval_episodes = 256
- seeds = {0,1,2,3,4}

## Repro commands

Run three experiment groups:

```bash
# 1) harder budget sweep (5 seeds)
SEEDS="0 1 2 3 4" \
ENV=v3 SPEND_MODE=adaptive \
NUM_CRITICAL_NODES=4 STOCHASTICITY=0.25 TRANSITION_NOISE=0.10 \
BUDGETS="0 1 2 4 8" \
bash experiments/exp_budget_sweep.sh

# 2) distillation ablation (5 seeds)
SEEDS="0 1 2 3 4" \
ENV=v3 SPEND_MODE=adaptive \
NUM_CRITICAL_NODES=4 STOCHASTICITY=0.25 TRANSITION_NOISE=0.10 \
B=4 KL_BC=0.0 KL_BC_KL=0.5 \
bash experiments/exp_distill_ablation.sh

# 3) offline vs rounds (5 seeds)
SEEDS="0 1 2 3 4" \
ENV=v3 SPEND_MODE=adaptive \
NUM_CRITICAL_NODES=4 STOCHASTICITY=0.25 TRANSITION_NOISE=0.10 \
B=4 KL_COEFF=0.0 \
ROUNDS=10 NUM_STEPS_PER_ROUND=200 EPISODES_PER_ROUND=64 \
bash experiments/exp_offline_vs_rounds.sh
```

Aggregate flat + seed stats:

```bash
python experiments/aggregate_results.py results/ \
  --csv results/all.csv --markdown results/all.md

python experiments/aggregate_results.py results/ \
  --group-by experiment,mode,budget,kl_coeff,spend_mode,env \
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

## Ceiling-effect sanity check

The aggregator now emits a warning if all rows have success_rate >= 0.98.
If warning appears, difficulty tuning failed and env knobs should be raised:
- increase `num_critical_nodes`
- increase `stochasticity`
- increase `transition_noise`
- require more `required_critical_passes`

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
- `results/budget_sweep/summary_seed_stats.{csv,md}`
- `results/distill_ablation/summary_seed_stats.{csv,md}`
- `results/offline_vs_rounds/summary_seed_stats.{csv,md}`
- `plots/success_vs_budget.png`
- `plots/cost_per_uplift.png`
- `plots/distill_ablation.png`
- `plots/offline_vs_rounds.png`
