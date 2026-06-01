"""Task success-rate evaluation.

Runs the (optionally trained) student in the toy env *without* teacher
intervention and reports the success rate over ``num_episodes``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from envs.factory import make_env  # noqa: E402
from models.policy_heads import MLPPolicy  # noqa: E402


def evaluate(
    checkpoint: str,
    num_episodes: int,
    horizon: int,
    num_actions: int,
    seed: int,
    greedy: bool,
    env_name: str = "v1",
    episode_steps: int | None = None,
    num_critical_nodes: int | None = None,
    stochasticity: float | None = None,
    transition_noise: float | None = None,
    required_critical_passes: int | None = None,
    difficulty: str | None = None,
    intervention_effect_span: int | None = None,
    failure_softness: str | None = None,
    allow_random_init: bool = False,
) -> dict:
    torch.manual_seed(seed)
    env_kwargs: dict = {}
    if env_name in {"v2", "v3"} and episode_steps is not None:
        env_kwargs["episode_steps"] = int(episode_steps)
    if env_name == "v3":
        if difficulty is not None:
            env_kwargs["difficulty"] = str(difficulty)
        if num_critical_nodes is not None:
            env_kwargs["num_critical_nodes"] = int(num_critical_nodes)
        if stochasticity is not None:
            env_kwargs["stochasticity"] = float(stochasticity)
        if transition_noise is not None:
            env_kwargs["transition_noise"] = float(transition_noise)
        if intervention_effect_span is not None:
            env_kwargs["intervention_effect_span"] = int(intervention_effect_span)
        if failure_softness is not None:
            env_kwargs["failure_softness"] = str(failure_softness)
        if required_critical_passes is not None:
            env_kwargs["required_critical_passes"] = int(required_critical_passes)
    env = make_env(
        env_name, num_actions=num_actions, horizon=horizon, seed=seed,
        **env_kwargs,
    )
    student = MLPPolicy(obs_dim=env.obs_dim, num_actions=env.num_actions)
    if checkpoint:
        if not os.path.isfile(checkpoint):
            if allow_random_init:
                print(
                    f"[eval_success] checkpoint not found at {checkpoint}; "
                    "--allow-random-init is set, evaluating random init."
                )
            else:
                raise FileNotFoundError(
                    f"[eval_success] checkpoint not found: {checkpoint}. "
                    "Pass --allow-random-init to evaluate an untrained policy on purpose."
                )
        else:
            state = torch.load(checkpoint, map_location="cpu")
            # strict=True surfaces shape mismatches instead of silently loading
            # a stale checkpoint from a different env / num_actions / horizon.
            student.load_state_dict(state, strict=True)
            print(f"[eval_success] loaded {checkpoint}")
    else:
        if not allow_random_init:
            raise ValueError(
                "[eval_success] --checkpoint is empty. "
                "Pass --allow-random-init to evaluate an untrained policy on purpose."
            )
        print("[eval_success] no checkpoint provided; evaluating random init.")
    student.eval()

    successes = 0
    rewards = []
    lengths = []
    max_steps = getattr(env, "episode_steps", horizon)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = student.act(obs_t, greedy=greedy)
            obs, reward, done, info = env.step(action)
            total_r += float(reward)
            steps += 1
            if done and info.get("success"):
                successes += 1
        rewards.append(total_r)
        lengths.append(steps)

    report = {
        "num_episodes": num_episodes,
        "success_rate": successes / max(1, num_episodes),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "mean_length": sum(lengths) / max(1, len(lengths)),
        "greedy": greedy,
        "env": env_name,
        "difficulty": str(getattr(env, "difficulty", "") or "") or None,
    }
    print(f"[eval_success] {json.dumps(report, indent=2)}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/student.pt")
    p.add_argument("--num-episodes", type=int, default=128)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--env", type=str, default="v1", choices=["v1", "v2", "v3"])
    p.add_argument(
        "--episode-steps", type=int, default=None,
        help="V2/V3 only: cap on episode length; defaults to --horizon.",
    )
    p.add_argument("--num-critical-nodes", type=int, default=None)
    p.add_argument("--stochasticity", type=float, default=None)
    p.add_argument("--transition-noise", type=float, default=None)
    p.add_argument("--required-critical-passes", type=int, default=None)
    p.add_argument(
        "--difficulty", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="V3 only: named difficulty preset; must match training.",
    )
    p.add_argument("--intervention-effect-span", type=int, default=None)
    p.add_argument(
        "--failure-softness", type=str, default=None,
        choices=["high", "medium", "low"],
    )
    p.add_argument(
        "--allow-random-init", action="store_true",
        help="Permit eval against a freshly-initialised student (no checkpoint loaded).",
    )
    p.add_argument("--output", type=str, default="eval_success.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = evaluate(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        num_actions=args.num_actions,
        seed=args.seed,
        greedy=args.greedy,
        env_name=args.env,
        episode_steps=args.episode_steps,
        num_critical_nodes=args.num_critical_nodes,
        stochasticity=args.stochasticity,
        transition_noise=args.transition_noise,
        required_critical_passes=args.required_critical_passes,
        difficulty=args.difficulty,
        intervention_effect_span=args.intervention_effect_span,
        failure_softness=args.failure_softness,
        allow_random_init=args.allow_random_init,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
