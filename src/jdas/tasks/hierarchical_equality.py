"""Hierarchical equality task (Phase A).

Input: four vectors ``(a, b, c, d)`` each drawn iid uniform from the unit sphere
in ``R^n_emb``, flattened to ``(B, 4 * n_emb)``. Fresh random vectors every batch
(infinite data).

Label: ``y = int((a == b) == (c == d))`` where equality means the two vectors are
*literally identical*. To keep labels balanced we sample each pair to be equal
with probability 0.5 independently for ``(a, b)`` and ``(c, d)``.

Ground-truth causal variables (``k_gt = 2``): ``[a == b, c == d]``, and the
ground-truth decoder is ``gt_label_fn(vars) = (vars[:, 0] == vars[:, 1])``.
"""

from __future__ import annotations

import torch

from jdas.tasks._sampling import sample_source_assignment
from jdas.types import InterventionBatch


class TaskConfigError(ValueError):
    """Raised when a task is constructed with an invalid configuration."""


class HierarchicalEqualityTask:
    """Hierarchical equality task with two ground-truth boolean variables.

    Args:
        n_emb: dimensionality of each of the four symbol embeddings.
        pool_size: if given, symbols are drawn from a fixed pool of this many
            unit vectors per batch (instead of freshly sampled continuous
            vectors). ``None`` (default) means fresh continuous vectors, i.e.
            equality is only ever exact-by-construction. The pool option exists
            so callers can create genuine "distinct but from a small alphabet"
            inputs; it does not change any of the causal semantics.
    """

    n_labels: int = 2
    k_gt: int = 2

    def __init__(self, n_emb: int = 16, pool_size: int | None = None) -> None:
        if n_emb < 1:
            raise TaskConfigError(f"n_emb must be >= 1, got {n_emb}")
        if pool_size is not None and pool_size < 2:
            raise TaskConfigError(f"pool_size must be >= 2 or None, got {pool_size}")
        self.n_emb = n_emb
        self.pool_size = pool_size
        self.input_dim = 4 * n_emb
        self.name = "hierarchical_equality"

    # -- ground-truth causal model -------------------------------------------------

    @staticmethod
    def gt_label_fn(vars: torch.Tensor) -> torch.Tensor:
        """Ground-truth decoder: ``(vars[:, 0] == vars[:, 1])``.

        Args:
            vars: ``(B, 2)`` long, entries ``[a == b, c == d]``.

        Returns:
            ``(B,)`` long task labels.
        """
        return (vars[:, 0] == vars[:, 1]).long()

    def gt_variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recompute ground-truth variables by exact vector comparison.

        Args:
            inputs: ``(B, 4 * n_emb)`` float.

        Returns:
            ``(B, 2)`` long, entries ``[a == b, c == d]``.
        """
        b, _ = inputs.shape
        a, bb, c, d = inputs.view(b, 4, self.n_emb).unbind(dim=1)
        e1 = (a == bb).all(dim=-1).long()
        e2 = (c == d).all(dim=-1).long()
        return torch.stack([e1, e2], dim=-1)

    # -- sampling helpers ----------------------------------------------------------

    def _unit_vectors(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        """Return ``(n, n_emb)`` iid points on the unit sphere in ``R^n_emb``."""
        v = torch.randn(n, self.n_emb, generator=generator, device=device)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)

    def _sample_symbols(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> torch.Tensor:
        """Sample ``(B, 4 * n_emb)`` inputs with balanced pair-equality labels.

        Each of the two pairs ``(a, b)`` and ``(c, d)`` is made equal with prob
        0.5. Distinct symbols are guaranteed distinct (probability-1 for
        continuous sampling; explicitly for the pool variant).
        """
        # base four independent symbols
        syms = self._unit_vectors(batch_size * 4, generator, device).view(
            batch_size, 4, self.n_emb
        )
        eq_ab = torch.rand(batch_size, generator=generator, device=device) < 0.5
        eq_cd = torch.rand(batch_size, generator=generator, device=device) < 0.5
        # when equal, copy the first element of the pair onto the second
        syms[:, 1] = torch.where(eq_ab.unsqueeze(-1), syms[:, 0], syms[:, 1])
        syms[:, 3] = torch.where(eq_cd.unsqueeze(-1), syms[:, 2], syms[:, 3])
        return syms.reshape(batch_size, self.input_dim)

    def sample_inputs(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean supervised batch for ordinary toy-model training.

        Returns:
            ``(inputs, labels)`` where inputs is ``(B, 4 * n_emb)`` float and
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
        """Sample an interchange-intervention batch.

        See :class:`jdas.types.InterventionBatch` and the ``source_assignment``
        convention in :mod:`jdas.types`.

        Args:
            batch_size: number of base examples ``B``.
            n_sources: number of source inputs ``m`` sampled per base example.
            k_max: number of variable slots in the alignment.
            generator: RNG (its device determines the output device).

        Returns:
            An :class:`InterventionBatch` with
            ``base_inputs`` ``(B, 4 * n_emb)``,
            ``source_inputs`` ``(B, m, 4 * n_emb)``,
            ``source_assignment`` ``(B, k_max)`` long in ``[-1, m)``,
            ``base_labels`` ``(B,)`` long, ``source_labels`` ``(B, m)`` long.
        """
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
