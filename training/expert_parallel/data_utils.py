"""Synthetic data generation and model configuration utilities for AutoEP example."""

from dataclasses import dataclass

import torch
from transformers import MixtralConfig


@dataclass
class SyntheticBatch:
    """A single synthetic batch of token IDs."""

    input_ids: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    attention_mask: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long
    labels: torch.Tensor  # [micro_batch_size, seq_len], dtype=torch.long (= input_ids)


class SyntheticBatchGenerator:
    """Generates deterministic synthetic token batches on demand.

    Batches are sharded by data-parallel rank to preserve distributed semantics.
    Uses a global microstep cursor (optimizer_step, accum_idx, dp_rank) to ensure
    both modes consume the same token order.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        micro_batch_size: int,
        total_steps: int,
        grad_accum: int,
        dp_world_size: int,
        dp_rank: int,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.total_steps = total_steps
        self.grad_accum = grad_accum
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.seed = seed

    def get_batch(self, optimizer_step: int, accum_idx: int) -> SyntheticBatch:
        """Return the batch for a specific (optimizer_step, accum_idx) coordinate.

        Uses dp_rank from __init__ to produce rank-specific sharding.
        Deterministic: same args always return same batch.

        Batch indexing formula:
            global_batch_idx = optimizer_step * grad_accum * dp_world_size
                             + accum_idx * dp_world_size + dp_rank
        Each global_batch_idx seeds a torch.Generator via:
            gen = torch.Generator().manual_seed(self.seed + global_batch_idx)
        """
        if optimizer_step < 0 or optimizer_step >= self.total_steps:
            raise ValueError(
                f"optimizer_step {optimizer_step} outside [0, {self.total_steps})"
            )
        if accum_idx < 0 or accum_idx >= self.grad_accum:
            raise ValueError(
                f"accum_idx {accum_idx} outside [0, {self.grad_accum})"
            )

        global_batch_idx = (
            optimizer_step * self.grad_accum * self.dp_world_size
            + accum_idx * self.dp_world_size
            + self.dp_rank
        )
        gen = torch.Generator().manual_seed(self.seed + global_batch_idx)

        input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.micro_batch_size, self.seq_len),
            generator=gen,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        return SyntheticBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def build_mixtral_config(
    num_layers: int,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
    output_router_logits: bool = False,
    attention_dropout: float = 0.0,
    router_jitter_noise: float = 0.0,
) -> MixtralConfig:
    """Build a MixtralConfig for random-weight initialization.

    Returns a config suitable for AutoModelForCausalLM.from_config().
    Sets max_position_embeddings = 4096 (Mixtral default).
    The caller must ensure max_position_embeddings >= seq_len used in training.
    """
    return MixtralConfig(
        num_hidden_layers=num_layers,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        output_router_logits=output_router_logits,
        attention_dropout=attention_dropout,
        router_jitter_noise=router_jitter_noise,
    )
