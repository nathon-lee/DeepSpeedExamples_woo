"""Online policy distillation trainer (DeepSpeed-optional).

The trainer consumes a JSONL of trajectories, extracts
``DistillationSample`` records keyed off intervention events, fills a
``ReplayBuffer``, and trains the student policy with a placeholder
behavior-cloning + KL distillation loss.

DeepSpeed is used to initialize the student if available; otherwise we fall
back to vanilla PyTorch so the example still runs on a single CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from data.intervention_dataset import load_distillation_samples  # noqa: E402
from data.replay_buffer import ReplayBuffer  # noqa: E402
from data.schema import DistillationSample  # noqa: E402
from envs.toy_long_horizon_env import ToyLongHorizonEnv  # noqa: E402
from models.policy_heads import MLPPolicy  # noqa: E402

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
    kl_coeff: float = 0.5,
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
) -> None:
    torch.manual_seed(seed)

    samples = load_distillation_samples(rollouts_path)
    if not samples:
        raise RuntimeError(
            f"No distillation samples found in {rollouts_path}. "
            "Did the collector produce any interventions?"
        )

    buffer = ReplayBuffer(
        capacity=capacity, prioritized=prioritized, seed=seed
    )
    buffer.extend(samples)
    print(f"[train] loaded {len(samples)} samples into replay buffer "
          f"(capacity={capacity}, prioritized={prioritized})")

    student = MLPPolicy(obs_dim=obs_dim, num_actions=num_actions)

    engine = None
    optimizer = None
    if _HAS_DEEPSPEED and ds_config and os.path.isfile(ds_config):
        # Minimal DeepSpeed init; works on CPU when no CUDA is available
        # because zero stage 0 is selected in the default config.
        try:
            engine, optimizer, _, _ = deepspeed.initialize(  # type: ignore[attr-defined]
                model=student,
                model_parameters=list(student.parameters()),
                config=ds_config,
            )
            print("[train] DeepSpeed engine initialized.")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[train] DeepSpeed init failed ({exc}); using vanilla PyTorch.")
            engine = None
            optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    else:
        if not _HAS_DEEPSPEED:
            print("[train] DeepSpeed not available; using vanilla PyTorch.")
        else:
            print(f"[train] DeepSpeed config not found at {ds_config}; "
                  "using vanilla PyTorch.")
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)

    student.train()
    history = []
    if engine is not None:
        device = engine.device
    else:
        device = next(student.parameters()).device
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
            loss = distill_loss(logits, tgt, teacher_logits, weights)
            engine.backward(loss)
            engine.step()
        else:
            optimizer.zero_grad()
            logits = student(obs)
            loss = distill_loss(logits, tgt, teacher_logits, weights)
            loss.backward()
            optimizer.step()

        if step % log_every == 0 or step == 1:
            print(f"[train] step={step}/{num_steps} loss={float(loss.item()):.4f}")
            history.append({"step": step, "loss": float(loss.item())})

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "student.pt")
    cpu_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save(cpu_state, ckpt_path)
    with open(os.path.join(output_dir, "train_history.json"), "w", encoding="utf-8") as f:
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
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=4)
    # Let DeepSpeed swallow its own flags when launched via `deepspeed`.
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    probe_env = ToyLongHorizonEnv(num_actions=args.num_actions, horizon=args.horizon)
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
    )
