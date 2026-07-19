"""Shared contracts between jdas core, tasks, and models.

Conventions used throughout:
- k_max: max number of high-level variables in the learned causal model.
- m: number of source inputs sampled per base example.
- An intervention for one example is a vector ``source_assignment`` of shape
  (k_max,) with integer entries in [-1, m): entry i == -1 means "do not swap
  variable i"; entry i == j >= 0 means "take variable i (and its aligned
  subspace) from source j". Multi-variable interventions use distinct j per
  swapped variable when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class InterventionBatch:
    """One training/eval batch of interchange interventions.

    Shapes (B = batch, m = n_sources, k = k_max):
    - base_inputs: task-specific tensor, e.g. (B, ...) floats or token ids.
    - source_inputs: (B, m, ...) same trailing shape as base_inputs.
    - source_assignment: (B, k) long in [-1, m).
    - base_labels: (B,) long — true task label of the base input.
    - source_labels: (B, m) long.
    """

    base_inputs: torch.Tensor
    source_inputs: torch.Tensor
    source_assignment: torch.Tensor
    base_labels: torch.Tensor
    source_labels: torch.Tensor


@runtime_checkable
class Task(Protocol):
    """A task with (optionally) known ground-truth causal variables."""

    n_labels: int
    k_gt: int  # number of ground-truth high-level variables (0 if unknown)

    def sample_batch(
        self, batch_size: int, n_sources: int, k_max: int, generator: torch.Generator
    ) -> InterventionBatch: ...

    def gt_variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, k_gt) long — ground-truth variable values, for recovery eval only."""
        ...


@runtime_checkable
class InterventionSite(Protocol):
    """A frozen network with one designated hidden-vector intervention site.

    Implementations wrap a toy MLP or an HF transformer layer/position.
    """

    d: int  # dimensionality of the hidden vector at the site
    n_labels: int

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, d) hidden vector at the site (no grad to weights; graph kept)."""
        ...

    def logits_with_hidden(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """(B, n_labels) — rerun from the site with `hidden` substituted."""
        ...

    def logits(self, inputs: torch.Tensor) -> torch.Tensor: ...
