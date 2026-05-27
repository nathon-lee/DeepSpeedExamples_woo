# DeepSpeed-SteerableAgents

MVP research prototype for the paper direction:

> **Budgeted Human Steering for Long-Horizon Agents via Online Policy Distillation**

This example provides a clean, runnable scaffold for studying how a long-horizon
agent (the *student*) can be improved by a small, *budgeted* number of
intervention events from a teacher/human (the *steerer*). Student improvement is
performed by **online policy distillation** from intervention-augmented
trajectories collected into a replay buffer.

This is a research prototype, **not** a polished product. The toy environment
and toy policy model exist purely to make the training/eval skeleton runnable.

---

## Problem being modeled

A student policy interacts with a long-horizon environment. Sometimes it is
uncertain or about to make a costly mistake. A teacher/human can intervene with
a structured event (progress update, plan correction, action veto, goal
redirect, ...). Intervention is **expensive** and limited by a per-episode and
per-experiment **budget**.

We want to:

1. Decide *when* to spend a steering budget unit (budget controller + teacher
   query policy).
2. Capture each intervention as supervision in a replay buffer.
3. **Distill** that supervision online into the student policy.
4. Evaluate task success, intervention usage, and basic steerability metrics.

---

## Directory structure

```
applications/DeepSpeed-SteerableAgents/
  README.md
  configs/
    student_ds_config.json        # DeepSpeed config for student training
    distill_ds_config.json        # DeepSpeed config for distillation runs
  data/
    schema.py                     # Trajectory & intervention dataclasses
    intervention_dataset.py       # Torch Dataset over collected rollouts
    replay_buffer.py              # Intervention/distillation replay buffer
  envs/
    base_env.py                   # Minimal env interface
    toy_long_horizon_env.py       # Toy multi-step env with failure modes
    toy_long_horizon_env_v2.py    # Learnable variant (good/trap fixed per ep.)
    factory.py                    # `make_env(name, ...)` switch
  models/
    policy_heads.py               # Toy student policy (MLP) + teacher stub
  training/
    budget_controller.py          # Decides whether to spend intervention budget
    teacher_query_policy.py       # Generates simulated teacher interventions
    online_distill_trainer.py     # DeepSpeed distillation training loop
    collect_rollouts.py           # Rollout collection with steering
  eval/
    eval_success.py               # Task success rate
    eval_steerability.py          # Steerability metrics
    eval_budget.py                # Budget consumption stats
    eval_uplift.py                # Cross-experiment uplift / cost-per-uplift
  scripts/
    run_collect.sh
    run_train.sh
    run_eval.sh
```

---

## Quick start

Requires Python 3.9+, PyTorch, and (optionally) DeepSpeed. The trainer falls
back to plain PyTorch if DeepSpeed is unavailable, so you can experiment
without a GPU cluster.

```bash
pip install -r requirements.txt
```

> **System dependencies (not pip-installable).** If you want to use DeepSpeed
> together with `mpi4py`, you must also install an MPI runtime, e.g. on
> Debian/Ubuntu:
>
> ```bash
> apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
> ```
>
> If you don't need MPI, simply skip `mpi4py`; DeepSpeed will run in
> single-process mode and the trainer will fall back to vanilla PyTorch on any
> DeepSpeed init failure.

```bash
# 1) Collect rollouts with budgeted teacher steering -> rollouts.jsonl
bash scripts/run_collect.sh

# 2) Train a student via online policy distillation on the replay buffer
bash scripts/run_train.sh

# 3) Run evaluation (success / steerability / budget)
bash scripts/run_eval.sh
```

All three scripts are thin wrappers around the corresponding Python modules and
can be edited/extended freely.

---

## Component summary

### Data schema (`data/schema.py`)
`TaskSpec`, `AgentState`, `ActionStep`, `ProgressUpdate`, `InterventionEvent`,
`Trajectory`, `DistillationSample`.

`InterventionEvent.kind` supports:
`progress_update`, `request_help`, `plan_correction`, `action_veto`,
`goal_redirect`.

### Toy long-horizon environment (`envs/`)

Two variants are provided; select with `--env v1|v2` (or `ENV=v2` for the
shell scripts).

- **`v1` — `ToyLongHorizonEnv`** (default). The hidden good/trap actions are
  resampled at *every* step and **not** revealed in the observation. The
  pipeline runs end-to-end, but the supervised target is essentially noise
  from the student's point of view, so distillation loss will plateau near
  `log(num_actions)`. Useful for sanity-checking that the framework runs.
- **`v2` — `ToyLongHorizonEnvV2`** (recommended for actual learning demos).
  The good and trap actions are fixed for the duration of an episode and
  exposed as one-hot fields inside the observation. With this env you should
  see loss decrease, success-rate climb, and a measurable uplift from
  steering.

Example end-to-end run with V2:

```bash
ENV=v2 HORIZON=16 NUM_ACTIONS=4 NUM_EPISODES=512 \
    GLOBAL_BUDGET=1024 PER_EP_BUDGET=4 THRESHOLD=0.4 \
    bash scripts/run_collect.sh

ENV=v2 HORIZON=16 NUM_ACTIONS=4 NUM_STEPS=3000 BATCH_SIZE=128 \
    bash scripts/run_train.sh

ENV=v2 HORIZON=16 NUM_ACTIONS=4 NUM_EVAL_EPISODES=256 \
    bash scripts/run_eval.sh
```

#### V2 timing knobs (`episode_steps` / `progress_goal`)

By default V2 requires the agent to accumulate `progress_goal == horizon`
within `episode_steps == horizon` steps, which makes every unsteered step
strictly fatal (any "wasted" step loses the episode) and produces a
*step-function* budget-vs-success curve. For paper-style smooth curves,
allow the episode to run longer than the progress goal:

```bash
# 8 "useful" actions needed, 24 steps allowed.
ENV=v2 HORIZON=8 NUM_ACTIONS=4 EPISODE_STEPS=24 \
    GLOBAL_BUDGET=256 PER_EP_BUDGET=4 THRESHOLD=0 \
    bash scripts/run_collect.sh
```

`EPISODE_STEPS` is plumbed through `run_collect.sh` / `run_train.sh` /
`run_eval.sh` and the corresponding `--episode-steps` flag on
`collect_rollouts.py` and `eval/eval_success.py`. See the next section
for a full sweep that actually trains a per-budget student.

#### Smooth budget sweep + uplift recipe

Run the full collect → train → eval pipeline across several budgets, then
post-aggregate with `eval/eval_uplift.py` for a per-budget uplift /
cost-per-uplift table. **Each budget must train its own student** —
without per-budget checkpoints every eval would reload the same model
and the table collapses to the baseline. The shell helpers resolve all
paths against the caller's cwd, so it's safe to run the loop from a
sub-directory:

```bash
cd applications/DeepSpeed-SteerableAgents
mkdir -p sweep && cd sweep

HORIZON=8
EPISODE_STEPS=24
ENV=v2
NUM_EPISODES=512
NUM_STEPS=2000
NUM_EVAL_EPISODES=256

for B in 0 1 2 3 4 6 8; do
    echo "=== budget=$B ==="
    # 1) collect rollouts at this budget
    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_EPISODES=$NUM_EPISODES \
        GLOBAL_BUDGET=$((B*NUM_EPISODES)) PER_EP_BUDGET=$B THRESHOLD=0 \
        OUTPUT=roll_b${B}.jsonl bash ../scripts/run_collect.sh

    # 2) train a per-budget student (B=0 just saves an untrained ckpt)
    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_STEPS=$NUM_STEPS BATCH_SIZE=128 \
        ROLLOUTS=roll_b${B}.jsonl \
        CHECKPOINT=ckpt_b${B}.pt bash ../scripts/run_train.sh

    # 3) eval that checkpoint; EVAL_PREFIX renames the three reports
    ENV=$ENV HORIZON=$HORIZON EPISODE_STEPS=$EPISODE_STEPS \
        NUM_EVAL_EPISODES=$NUM_EVAL_EPISODES \
        CHECKPOINT=ckpt_b${B}.pt \
        ROLLOUTS=roll_b${B}.jsonl \
        EVAL_PREFIX=b${B}_ bash ../scripts/run_eval.sh
done

python ../eval/eval_uplift.py \
    --baseline b0_eval_success.json \
    --run b1:b1_eval_success.json:b1_eval_budget.json \
    --run b2:b2_eval_success.json:b2_eval_budget.json \
    --run b3:b3_eval_success.json:b3_eval_budget.json \
    --run b4:b4_eval_success.json:b4_eval_budget.json \
    --run b6:b6_eval_success.json:b6_eval_budget.json \
    --run b8:b8_eval_success.json:b8_eval_budget.json \
    --output uplift.json
```

Sample output from the recipe above (numbers vary by seed):

```
baseline success_rate = 0.004
name         succ   uplift     n_iv   cost/+1%
b1          0.996   +0.992      512       5.16
b2          0.996   +0.992     1024      10.32
b3          0.996   +0.992     1536      15.48
b4          0.992   +0.988     2048      20.72
b6          0.988   +0.984     3072      31.21
b8          0.984   +0.980     4096      41.78
```

Reading this table: distillation converts even a single hint per episode
into near-perfect policy performance, so the curve **saturates at the
bottom** and `cost_per_uplift_point` rises monotonically — B=1 is the
sweet spot for this regime. To expose a fuller S-curve where small B
genuinely underperforms, make the task harder (larger `NUM_ACTIONS`,
larger `HORIZON`, fewer `NUM_EPISODES`) so a single hint no longer
suffices, e.g.:

```bash
HORIZON=16 NUM_ACTIONS=8 EPISODE_STEPS=48 \
NUM_EPISODES=128 NUM_STEPS=1500 BATCH_SIZE=64 \
NUM_EVAL_EPISODES=256 ENV=v2
# then sweep B in 0 1 2 4 6 8 10 12 14 16
```

### Budget controller (`training/budget_controller.py`)
Heuristic controller combining:
- predicted uncertainty (entropy of student logits),
- horizon progress (step index / max steps),
- remaining per-episode + global budget.

### Teacher query policy (`training/teacher_query_policy.py`)
Rule-based simulated teacher that emits the most useful `InterventionEvent`
given current env state.

### Replay buffer (`data/replay_buffer.py`)
Bounded buffer with optional prioritization by sample `quality` score.

### Online distillation trainer (`training/online_distill_trainer.py`)
- Initializes the student via DeepSpeed (if available).
- Loops over replay-buffer mini-batches.
- Computes a placeholder behavior-cloning + KL distillation loss on
  intervention-supervised steps.

### Evaluation hooks (`eval/`)
Independent scripts for success rate, intervention usage / budget, basic
steerability (avoided-bad-action rate, success uplift from intervention),
and a cross-experiment **uplift aggregator** (`eval_uplift.py`) that
joins a baseline `eval_success.json` with one or more steered
`(success, budget)` report pairs and reports
`success_uplift` and `cost_per_uplift_point` per run.

---

## What's implemented vs. future work

**Implemented (MVP):**
- Full data schema with serialization.
- Runnable toy long-horizon environment with intervention points.
- Budget controller + teacher query policy (rule-based).
- Replay buffer with prioritization.
- Online distillation trainer skeleton with optional DeepSpeed.
- Rollout collector that writes JSONL trajectories.
- Three eval scripts producing JSON reports.

**Future work (intentionally out of scope):**
- Real LLM-scale student / teacher policies.
- Real human feedback service (currently simulated).
- Distributed replay buffer / async collection.
- DeepSpeed core modifications, ZeRO-Offload tuning.
- Sophisticated steerability metrics (counterfactual evals, calibration, etc.).
- Web UI / labeling tool for live human steering.

---

## Assumptions

- The "teacher" is a deterministic oracle over the toy env; in real use it
  would be a stronger model or human-in-the-loop.
- Distillation loss is intentionally a simple BC + KL placeholder so it is easy
  to swap for the loss a real paper would use.
- DeepSpeed is optional: import errors fall back to vanilla PyTorch so the
  scaffold runs on a single CPU.
