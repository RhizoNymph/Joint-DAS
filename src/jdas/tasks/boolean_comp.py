"""Boolean composition task (Phase A).

Three binary inputs ``x1, x2, x3``. Each slot has *two* fixed random embedding
vectors (one for value 0, one for value 1), fixed at task construction with its
own seed. The input is the concatenation of the chosen embeddings, ``(B, 3 *
n_emb)``.

Label: ``y = (x1 AND x2) OR x3``.

Ground-truth causal variables (``k_gt = 2``): ``[x1 AND x2, x3]`` and the
ground-truth decoder is ``gt_label_fn(vars) = vars[:, 0] OR vars[:, 1]``.
"""

from __future__ import annotations

import torch

from jdas.tasks._sampling import sample_source_assignment
from jdas.tasks.hierarchical_equality import TaskConfigError
from jdas.types import InterventionBatch


class BooleanCompositionTask:
    """Boolean composition task ``y = (x1 & x2) | x3`` over fixed embeddings.

    Args:
        n_emb: dimensionality of each of the three slot embeddings.
        seed: seed for the fixed embedding table (so runs are reproducible).
    """

    n_labels: int = 2
    k_gt: int = 2

    def __init__(self, n_emb: int = 16, seed: int = 0) -> None:
        if n_emb < 1:
            raise TaskConfigError(f"n_emb must be >= 1, got {n_emb}")
        self.n_emb = n_emb
        self.seed = seed
        self.input_dim = 3 * n_emb
        self.name = "boolean_composition"
        # fixed embedding table: (3 slots, 2 values, n_emb), on the unit sphere.
        gen = torch.Generator().manual_seed(seed)
        table = torch.randn(3, 2, n_emb, generator=gen)
        self.embed = table / table.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(table.dtype).tiny
        )

    # -- ground-truth causal model -------------------------------------------------

    @staticmethod
    def gt_label_fn(vars: torch.Tensor) -> torch.Tensor:
        """Ground-truth decoder: ``vars[:, 0] OR vars[:, 1]``.

        Args:
            vars: ``(B, 2)`` long, entries ``[x1 AND x2, x3]``.

        Returns:
            ``(B,)`` long task labels.
        """
        return (vars[:, 0].bool() | vars[:, 1].bool()).long()

    def _bits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recover the three raw bits ``(B, 3)`` from embeddings by matching.

        Each slot's embedding equals one of two fixed vectors; identify which by
        exact comparison against the value-1 vector for that slot.
        """
        b, _ = inputs.shape
        chunks = inputs.view(b, 3, self.n_emb)
        embed = self.embed.to(inputs.device)
        # bit == 1 iff the slot vector matches the value-1 embedding exactly
        bits = (chunks == embed[:, 1].unsqueeze(0)).all(dim=-1).long()
        return bits

    def gt_variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recompute ground-truth variables from embeddings.

        Args:
            inputs: ``(B, 3 * n_emb)`` float.

        Returns:
            ``(B, 2)`` long, entries ``[x1 AND x2, x3]``.
        """
        bits = self._bits(inputs)
        x1_and_x2 = (bits[:, 0] & bits[:, 1]).long()
        x3 = bits[:, 2].long()
        return torch.stack([x1_and_x2, x3], dim=-1)

    # -- sampling helpers ----------------------------------------------------------

    def _embed_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 3)`` long bits to ``(B, 3 * n_emb)`` embeddings."""
        b = bits.shape[0]
        embed = self.embed.to(bits.device)  # (3, 2, n_emb)
        slot_ix = torch.arange(3, device=bits.device).unsqueeze(0).expand(b, 3)
        chosen = embed[slot_ix, bits]  # (B, 3, n_emb)
        return chosen.reshape(b, self.input_dim)

    def _sample_symbols(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        """Sample ``(B, 3 * n_emb)`` inputs with uniform random bits."""
        bits = (
            torch.rand(batch_size, 3, generator=generator, device=device) < 0.5
        ).long()
        return self._embed_bits(bits)

    def sample_inputs(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean supervised batch for ordinary toy-model training.

        Returns:
            ``(inputs, labels)`` where inputs is ``(B, 3 * n_emb)`` float and
            labels is ``(B,)`` long.
        """
        device = generator.device
        inputs = self._sample_symbols(batch_size, generator, device)
        labels = self.gt_label_fn(self.gt_variables(inputs))
        return inputs, labels

    def sample_batch(
        self,
        batch_size: int,
        n_sources: int,
        k_max: int,
        generator: torch.Generator,
    ) -> InterventionBatch:
        """Sample an interchange-intervention batch (see :class:`InterventionBatch`)."""
        device = generator.device
        base_inputs = self._sample_symbols(batch_size, generator, device)
        source_inputs = torch.stack(
            [
                self._sample_symbols(batch_size, generator, device)
                for _ in range(n_sources)
            ],
            dim=1,
        )
        base_labels = self.gt_label_fn(self.gt_variables(base_inputs))
        source_labels = torch.stack(
            [
                self.gt_label_fn(self.gt_variables(source_inputs[:, j]))
                for j in range(n_sources)
            ],
            dim=1,
        )
        source_assignment = sample_source_assignment(
            batch_size, n_sources, k_max, self.k_gt, generator, device
        )
        return InterventionBatch(
            base_inputs=base_inputs,
            source_inputs=source_inputs,
            source_assignment=source_assignment,
            base_labels=base_labels,
            source_labels=source_labels,
        )
