"""Factory for selecting a toy env variant by name.

Keeps a single point of switch so the collector / trainer / eval scripts
can all be parametrised the same way (``--env v1`` or ``--env v2``).

For V2, optional ``episode_steps`` and ``progress_goal`` can be supplied
via kwargs to enable the "budget-of-time" variant useful for smooth
budget-vs-success sweeps.
"""
from __future__ import annotations

from typing import Any, Optional

from .base_env import BaseEnv
from .toy_long_horizon_env import ToyLongHorizonEnv
from .toy_long_horizon_env_v2 import ToyLongHorizonEnvV2
from .toy_long_horizon_env_v3 import ToyLongHorizonEnvV3


def make_env(
    name: str,
    num_actions: int,
    horizon: int,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> BaseEnv:
    name = (name or "v1").lower()
    if name == "v1":
        return ToyLongHorizonEnv(
            num_actions=num_actions, horizon=horizon, seed=seed
        )
    if name == "v2":
        return ToyLongHorizonEnvV2(
            num_actions=num_actions, horizon=horizon, seed=seed, **kwargs
        )
    if name == "v3":
        return ToyLongHorizonEnvV3(
            num_actions=num_actions, horizon=horizon, seed=seed, **kwargs
        )
    raise ValueError(
        f"Unknown env name: {name!r} (expected 'v1', 'v2', or 'v3')"
    )
