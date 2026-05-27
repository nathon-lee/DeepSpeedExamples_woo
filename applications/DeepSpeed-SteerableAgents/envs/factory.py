"""Factory for selecting a toy env variant by name.

Keeps a single point of switch so the collector / trainer / eval scripts
can all be parametrised the same way (``--env v1`` or ``--env v2``).
"""
from __future__ import annotations

from typing import Optional

from .base_env import BaseEnv
from .toy_long_horizon_env import ToyLongHorizonEnv
from .toy_long_horizon_env_v2 import ToyLongHorizonEnvV2


def make_env(
    name: str,
    num_actions: int,
    horizon: int,
    seed: Optional[int] = None,
) -> BaseEnv:
    name = (name or "v1").lower()
    if name == "v1":
        return ToyLongHorizonEnv(
            num_actions=num_actions, horizon=horizon, seed=seed
        )
    if name == "v2":
        return ToyLongHorizonEnvV2(
            num_actions=num_actions, horizon=horizon, seed=seed
        )
    raise ValueError(f"Unknown env name: {name!r} (expected 'v1' or 'v2')")
