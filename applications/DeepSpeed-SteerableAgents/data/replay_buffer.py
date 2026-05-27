"""Bounded replay buffer for distillation samples.

Supports uniform or prioritized sampling by per-sample ``quality`` score.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Optional

from .schema import DistillationSample


class ReplayBuffer:
    """In-memory replay buffer for ``DistillationSample`` objects.

    Args:
        capacity: maximum number of samples kept; oldest are dropped (FIFO).
        prioritized: if True, ``sample`` draws proportional to ``quality``.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        prioritized: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.prioritized = bool(prioritized)
        self._buf: List[DistillationSample] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, sample: DistillationSample) -> None:
        if len(self._buf) >= self.capacity:
            self._buf.pop(0)
        self._buf.append(sample)

    def extend(self, samples: Iterable[DistillationSample]) -> None:
        for s in samples:
            self.add(s)

    def sample(self, batch_size: int) -> List[DistillationSample]:
        """Draw ``batch_size`` samples (with replacement)."""
        if not self._buf:
            return []
        if self.prioritized:
            weights = [max(1e-6, float(s.quality)) for s in self._buf]
            return self._rng.choices(self._buf, weights=weights, k=batch_size)
        return [self._rng.choice(self._buf) for _ in range(batch_size)]

    def clear(self) -> None:
        self._buf.clear()

    def all(self) -> List[DistillationSample]:
        return list(self._buf)
