"""Online policy distillation trainer (DeepSpeed-optional).

Two modes:

* **Offline (default):** consume a frozen JSONL of trajectories, extract
  ``DistillationSample`` records keyed off intervention events, fill a
  ``ReplayBuffer``, and train the student with CE + KL distillation loss.
* **Rounds / online (``--rounds N > 0``):** alternate ``collect ->
  extend replay buffer -> train K steps`` for ``N`` rounds, with the
  collector using the *current* student snapshot. This is what makes the
  "Online" in "Online Policy Distillation" defensible: the data
  distribution shifts as the student improves and the budget gate fires
  on whatever uncertainty the student currently has.

DeepSpeed is used to initialize the student if available; otherwise we
fall back to vanilla PyTorch so the example still runs on a single CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from data.intervention_dataset import (  # noqa: E402
    iter_distillation_samples,
    load_distillation_samples,
)
from data.replay_buffer import ReplayBuffer  # noqa: E402
from data.schema import DistillationSample  # noqa: E402
from envs.factory import make_env  # noqa: E402
from models.policy_heads import MLPPolicy  # noqa: E402
from training.collect_rollouts import collect  # noqa: E402

try:
    import deepspeed  # type: ignore

    _HAS_DEEPSPEED = True
except Exception:  # pragma: no cover - depends on local install
    deepspeed = None  # type: ignore
    _HAS_DEEPSPEED = False


def _batch_tensors(samples: List[DistillationSample], num_actions: int):
    obs = torch.tensor([s.observation for s in samples], dtype=torch.float32)
    tgt = torch.tensor([s.target_action for s in samples], dtype=torch.long)
    teacher_logits = None
    if all(s.teacher_logits is not None for s in samples):
        teacher_logits = torch.tensor(
            [s.teacher_logits for s in samples], dtype=torch.float32
        )
    weights = torch.tensor([s.quality for s in samples], dtype=torch.float32)
    return obs, tgt, teacher_logits, weights


def distill_loss(
    student_logits: torch.Tensor,
    target_actions: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    weights: torch.Tensor,
    kl_coeff: float = 0.0,
) -> torch.Tensor:
    """Behavior-cloning CE + optional KL to teacher distribution."""
    ce = F.cross_entropy(student_logits, target_actions, reduction="none")
    loss = (ce * weights).mean()
    if teacher_logits is not None:
        log_p_student = F.log_softmax(student_logits, dim=-1)
        p_teacher = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)
        loss = loss + kl_coeff * (kl * weights).mean()
    return loss


def _init_optimizer(student: torch.nn.Module, ds_config: str):
    """Initialise either a DeepSpeed engine or a vanilla AdamW optimizer.

    Returns ``(engine_or_None, optimizer_or_None, device)``.
    """
    if _HAS_DEEPSPEED and ds_config and os.path.isfile(ds_config):
        try:
            engine, optimizer, _, _ = deepspeed.initialize(  # type: ignore[attr-defined]
                model=student,
                model_parameters=list(student.parameters()),
                config=ds_config,
            )
            print("[train] DeepSpeed engine initialized.")
            return engine, optimizer, engine.device
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[train] DeepSpeed init failed ({exc}); using vanilla PyTorch.")
    elif not _HAS_DEEPSPEED:
        print("[train] DeepSpeed not available; using vanilla PyTorch.")
    else:
        print(f"[train] DeepSpeed config not found at {ds_config}; "
              "using vanilla PyTorch.")
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    return None, optimizer, next(student.parameters()).device


def _train_steps(
    student: torch.nn.Module,
    engine,
    optimizer,
    device,
    buffer: ReplayBuffer,
    num_steps: int,
    batch_size: int,
    num_actions: int,
    log_every: int,
    history: list,
    log_prefix: str = "train",
    global_step_offset: int = 0,
    kl_coeff: float = 0.0,
) -> int:
    """Run ``num_steps`` of distillation updates on samples in ``buffer``.

    Returns the new global step counter (offset + num_steps).
    """
    if len(buffer) == 0:
        print(f"[{log_prefix}] buffer empty; skipping {num_steps} update steps.")
        return global_step_offset
    student.train()
    for step in range(1, num_steps + 1):
        batch = buffer.sample(batch_size)
        obs, tgt, teacher_logits, weights = _batch_tensors(batch, num_actions)
        obs = obs.to(device)
        tgt = tgt.to(device)
        weights = weights.to(device)
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(device)

        if engine is not None:
            logits = engine(obs)
            loss = distill_loss(
                logits, tgt, teacher_logits, weights, kl_coeff=kl_coeff,
            )
            engine.backward(loss)
            engine.step()
        else:
            optimizer.zero_grad()
            logits = student(obs)
            loss = distill_loss(
                logits, tgt, teacher_logits, weights, kl_coeff=kl_coeff,
            )
            loss.backward()
            optimizer.step()

        g_step = global_step_offset + step
        if g_step % log_every == 0 or step == 1:
            kl_active = teacher_logits is not None and kl_coeff > 0
            print(
                f"[{log_prefix}] step={g_step} loss={float(loss.item()):.4f}"
                f" kl={'on' if kl_active else 'off'}"
            )
            history.append({"step": g_step, "loss": float(loss.item())})
    return global_step_offset + num_steps


def train(
    rollouts_path: str,
    ds_config: str,
    batch_size: int,
    num_steps: int,
    capacity: int,
    prioritized: bool,
    seed: int,
    output_dir: str,
    num_actions: int,
    obs_dim: int,
    log_every: int = 50,
    checkpoint_path: str = "",
    kl_coeff: float = 0.0,
) -> None:
    """Offline path: load JSONL once, train ``num_steps`` updates."""
    torch.manual_seed(seed)

    samples = load_distillation_samples(rollouts_path)
    if not samples:
        # For a B=0 baseline run there are legitimately no interventions to
        # distil from. Don't error out -- just skip the loop and save the
        # (randomly initialised) student so downstream eval has a checkpoint.
        print(
            f"[train] WARNING: no distillation samples in {rollouts_path}; "
            "saving an untrained checkpoint for baseline comparison."
        )
        student = MLPPolicy(obs_dim=obs_dim, num_actions=num_actions)
        _save_checkpoint(student, output_dir, checkpoint_path, history=[])
        return

    # Fail-fast sanity check: obs_dim derived from CLI/env must match the
    # observation length actually stored in the rollouts file.
    actual_obs_dim = len(samples[0].observation)
    if actual_obs_dim != obs_dim:
        raise ValueError(
            f"[train] obs_dim mismatch: env says {obs_dim} but rollouts file "
            f"{rollouts_path} has observations of length {actual_obs_dim}. "
            "Did you change --env / --num-actions / --horizon between collect "
            "and train?"
        )

    buffer = ReplayBuffer(
        capacity=capacity, prioritized=prioritized, seed=seed
    )
    buffer.extend(samples)
    print(f"[train] loaded {len(samples)} samples into replay buffer "
          f"(capacity={capacity}, prioritized={prioritized})")

    student = MLPPolicy(obs_dim=obs_dim, num_actions=num_actions)
    engine, optimizer, device = _init_optimizer(student, ds_config)
    history: list = []
    _train_steps(
        student=student, engine=engine, optimizer=optimizer, device=device,
        buffer=buffer, num_steps=num_steps, batch_size=batch_size,
        num_actions=num_actions, log_every=log_every, history=history,
        kl_coeff=kl_coeff,
    )
    _save_checkpoint(student, output_dir, checkpoint_path, history=history)


def train_rounds(
    rounds: int,
    episodes_per_round: int,
    num_steps_per_round: int,
    env_name: str,
    horizon: int,
    num_actions: int,
    episode_steps: Optional[int],
    global_budget_per_round: int,
    per_episode_budget: int,
    threshold: float,
    spend_mode: str,
    num_critical_nodes: int,
    stochasticity: float,
    transition_noise: float,
    required_critical_passes: Optional[int],
    ds_config: str,
    batch_size: int,
    capacity: int,
    prioritized: bool,
    seed: int,
    output_dir: str,
    checkpoint_path: str,
    log_every: int = 50,
    seed_rollouts_path: str = "",
    kl_coeff: float = 0.0,
) -> None:
    """Streaming / online path: alternate collect -> extend -> train K steps."""
    torch.manual_seed(seed)

    # Build a probe env once to lock obs_dim / num_actions for the student.
    env_kwargs = {}
    if env_name in {"v2", "v3"} and episode_steps is not None:
        env_kwargs["episode_steps"] = int(episode_steps)
    if env_name == "v3":
        env_kwargs["num_critical_nodes"] = int(num_critical_nodes)
        env_kwargs["stochasticity"] = float(stochasticity)
        env_kwargs["transition_noise"] = float(transition_noise)
        if required_critical_passes is not None:
            env_kwargs["required_critical_passes"] = int(required_critical_passes)
    probe_env = make_env(
        env_name, num_actions=num_actions, horizon=horizon, seed=seed,
        **env_kwargs,
    )
    obs_dim = probe_env.obs_dim

    student = MLPPolicy(obs_dim=obs_dim, num_actions=num_actions)
    engine, optimizer, device = _init_optimizer(student, ds_config)

    buffer = ReplayBuffer(
        capacity=capacity, prioritized=prioritized, seed=seed
    )

    # Optional warm-start: pre-populate the buffer from a previous offline run.
    if seed_rollouts_path and os.path.isfile(seed_rollouts_path):
        seed_samples = load_distillation_samples(seed_rollouts_path)
        if seed_samples:
            actual_obs_dim = len(seed_samples[0].observation)
            if actual_obs_dim != obs_dim:
                raise ValueError(
                    f"[train-rounds] seed rollouts obs_dim {actual_obs_dim} "
                    f"!= env obs_dim {obs_dim}; refusing to mix."
                )
            buffer.extend(seed_samples)
            print(f"[train-rounds] seeded buffer with {len(seed_samples)} samples")

    # Where to drop per-round artefacts.
    art_dir = os.path.dirname(os.path.abspath(checkpoint_path)) if checkpoint_path \
        else output_dir
    os.makedirs(art_dir, exist_ok=True)

    history: list = []
    global_step = 0
    for r in range(1, rounds + 1):
        # Collect with the *current* student (in-memory; no checkpoint I/O).
        rollouts_out = os.path.join(art_dir, f"_round_{r}_rollouts.jsonl")
        trajs = collect(
            num_episodes=episodes_per_round,
            horizon=horizon,
            num_actions=num_actions,
            global_budget=global_budget_per_round,
            per_episode_budget=per_episode_budget,
            threshold=threshold,
            seed=seed + r,
            output_path=rollouts_out,
            env_name=env_name,
            episode_steps=episode_steps,
            spend_mode=spend_mode,
            num_critical_nodes=num_critical_nodes,
            stochasticity=stochasticity,
            transition_noise=transition_noise,
            required_critical_passes=required_critical_passes,
            student=student,
        )

        new_samples: list = []
        for t in trajs:
            new_samples.extend(iter_distillation_samples(t))
        buffer.extend(new_samples)
        n_succ = sum(1 for t in trajs if t.success)
        print(
            f"[train-rounds] round={r}/{rounds} collected ep={len(trajs)} "
            f"success={n_succ}/{len(trajs)} new_samples={len(new_samples)} "
            f"buf={len(buffer)}"
        )

        global_step = _train_steps(
            student=student, engine=engine, optimizer=optimizer, device=device,
            buffer=buffer, num_steps=num_steps_per_round,
            batch_size=batch_size, num_actions=num_actions,
            log_every=log_every, history=history,
            log_prefix=f"train-rounds.r{r}", global_step_offset=global_step,
            kl_coeff=kl_coeff,
        )

    _save_checkpoint(student, output_dir, checkpoint_path, history=history)


def _save_checkpoint(
    student: torch.nn.Module,
    output_dir: str,
    checkpoint_path: str,
    history: list,
) -> None:
    if checkpoint_path:
        ckpt_path = checkpoint_path
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path)) or "."
    else:
        ckpt_dir = output_dir
        ckpt_path = os.path.join(ckpt_dir, "student.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save(cpu_state, ckpt_path)
    with open(os.path.join(ckpt_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] saved checkpoint -> {ckpt_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online distillation training.")
    p.add_argument("--rollouts", type=str, default="rollouts.jsonl")
    p.add_argument(
        "--ds-config",
        type=str,
        default=os.path.join(APP_DIR, "configs", "distill_ds_config.json"),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--capacity", type=int, default=100_000)
    p.add_argument("--prioritized", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Full path to write the checkpoint to. "
             "Overrides --output-dir/student.pt when given.",
    )
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=4)
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v2", "v3"])
    p.add_argument(
        "--episode-steps", type=int, default=None,
        help="V2/V3 only: cap on episode length (used by rounds mode).",
    )
    p.add_argument(
        "--spend-mode", type=str, default="adaptive",
        choices=["forced", "adaptive"],
        help="Intervention spend policy for rounds-mode collection.",
    )
    p.add_argument("--num-critical-nodes", type=int, default=4)
    p.add_argument("--stochasticity", type=float, default=0.25)
    p.add_argument("--transition-noise", type=float, default=0.10)
    p.add_argument("--required-critical-passes", type=int, default=None)

    # ---- Rounds / online mode -------------------------------------------
    p.add_argument(
        "--rounds", type=int, default=0,
        help="If > 0, run streaming collect->extend->train for this many "
             "rounds instead of the offline JSONL path.",
    )
    p.add_argument("--episodes-per-round", type=int, default=64)
    p.add_argument(
        "--num-steps-per-round", type=int, default=200,
        help="Optimisation steps per round in rounds mode.",
    )
    p.add_argument(
        "--global-budget-per-round", type=int, default=128,
        help="Intervention budget *per round* (collector resets per round).",
    )
    p.add_argument("--per-episode-budget", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument(
        "--seed-rollouts", type=str, default="",
        help="Optional path to a previously collected JSONL used to "
             "warm-start the replay buffer in rounds mode.",
    )
    p.add_argument(
        "--kl-coeff", type=float, default=0.0,
        help="Weight on the KL-to-teacher term. Default 0.0 (BC-only). "
             "Set >0 (e.g. 0.5) to enable KL distillation; in our toy v2 "
             "experiments with peaked teacher logits, kl_coeff=0 beats "
             "kl_coeff=0.5 by ~2.6pp (p<0.001 across 5 seeds), so KL is "
             "off by default. Re-enable on environments where the teacher "
             "produces a softer / more informative distribution.",
    )

    # Let DeepSpeed swallow its own flags when launched via `deepspeed`.
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    probe_env_kwargs = {}
    if args.env in {"v2", "v3"} and args.episode_steps is not None:
        probe_env_kwargs["episode_steps"] = args.episode_steps
    if args.env == "v3":
        probe_env_kwargs["num_critical_nodes"] = args.num_critical_nodes
        probe_env_kwargs["stochasticity"] = args.stochasticity
        probe_env_kwargs["transition_noise"] = args.transition_noise
        if args.required_critical_passes is not None:
            probe_env_kwargs["required_critical_passes"] = args.required_critical_passes
    probe_env = make_env(
        args.env,
        num_actions=args.num_actions,
        horizon=args.horizon,
        **probe_env_kwargs,
    )
    if args.rounds > 0:
        print(f"[train] rounds mode: rounds={args.rounds} "
              f"episodes_per_round={args.episodes_per_round} "
              f"steps_per_round={args.num_steps_per_round}")
        train_rounds(
            rounds=args.rounds,
            episodes_per_round=args.episodes_per_round,
            num_steps_per_round=args.num_steps_per_round,
            env_name=args.env,
            horizon=args.horizon,
            num_actions=probe_env.num_actions,
            episode_steps=args.episode_steps,
            global_budget_per_round=args.global_budget_per_round,
            per_episode_budget=args.per_episode_budget,
            threshold=args.threshold,
            spend_mode=args.spend_mode,
            num_critical_nodes=args.num_critical_nodes,
            stochasticity=args.stochasticity,
            transition_noise=args.transition_noise,
            required_critical_passes=args.required_critical_passes,
            ds_config=args.ds_config,
            batch_size=args.batch_size,
            capacity=args.capacity,
            prioritized=args.prioritized,
            seed=args.seed,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            seed_rollouts_path=args.seed_rollouts,
            kl_coeff=args.kl_coeff,
        )
    else:
        train(
            rollouts_path=args.rollouts,
            ds_config=args.ds_config,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            capacity=args.capacity,
            prioritized=args.prioritized,
            seed=args.seed,
            output_dir=args.output_dir,
            num_actions=probe_env.num_actions,
            obs_dim=probe_env.obs_dim,
            checkpoint_path=args.checkpoint_path,
            kl_coeff=args.kl_coeff,
        )
