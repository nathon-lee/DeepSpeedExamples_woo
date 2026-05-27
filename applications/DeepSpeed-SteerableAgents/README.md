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

### Toy long-horizon environment (`envs/toy_long_horizon_env.py`)
A deterministic-ish grid-of-decisions environment: at each step the agent picks
one of `K` actions; a hidden "good action" advances progress; some actions are
*trap* actions that fail the episode. Episodes are long enough (default 32
steps) to be "long-horizon" in spirit. Supports intervention points after every
step.

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
Independent scripts for success rate, intervention usage / budget, and basic
steerability (avoided-bad-action rate, success uplift from intervention).

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
