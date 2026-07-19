"""Tiny inline fake Task and InterventionSite implementations for core tests.

These deliberately avoid importing ``jdas.tasks`` / ``jdas.models`` (authored by
a sibling agent) so the core tests are self-contained.
"""

from __future__ import annotations

import torch
from torch import nn

from jdas.types import InterventionBatch


class XorTask:
    """Two binary GT variables z0, z1 in R^{2*emb}; label = z0 XOR z1.

    Input is a (B, 2, emb) tensor.  Variable j is 1 iff the mean of block j is
    positive.  Ground-truth label = z0 XOR z1.
    """

    def __init__(self, emb: int = 3) -> None:
        self.emb = emb
        self.n_labels = 2
        self.k_gt = 2

    def _bits(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(inputs.shape[0], 2, self.emb)
        return (x.mean(dim=-1) > 0).long()  # (B, 2)

    def gt_variables(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._bits(inputs)

    def label_from_variables(self, vars: torch.Tensor) -> torch.Tensor:
        return (vars[:, 0] ^ vars[:, 1]).long()

    def _labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.label_from_variables(self._bits(inputs))

    def sample_batch(
        self, batch_size: int, n_sources: int, k_max: int, generator: torch.Generator
    ) -> InterventionBatch:
        shape = (batch_size, 2, self.emb)
        base = torch.randn(shape, generator=generator)
        source = torch.randn((batch_size, n_sources, 2, self.emb), generator=generator)
        assign = torch.full((batch_size, k_max), -1, dtype=torch.long)
        return InterventionBatch(
            base_inputs=base,
            source_inputs=source,
            source_assignment=assign,
            base_labels=self._labels(base),
            source_labels=self._labels(source.reshape(-1, 2, self.emb)).reshape(
                batch_size, n_sources
            ),
        )


class MLPSite(nn.Module):
    """A frozen 2-layer MLP intervention site at its hidden layer.

    ``hidden`` returns the post-activation hidden of the first layer; the head
    maps that hidden to logits.  Weights are frozen (no grad) but the graph from
    a substituted hidden to logits is preserved.
    """

    def __init__(self, in_dim: int, d: int, n_labels: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.n_labels = n_labels
        self.enc = nn.Linear(in_dim, d)
        self.head = nn.Linear(d, n_labels)
        for lin in (self.enc, self.head):
            nn.init.normal_(lin.weight, generator=g)
            nn.init.zeros_(lin.bias)
        for p in self.parameters():
            p.requires_grad_(False)

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(inputs.shape[0], -1)
        return torch.relu(self.enc(x))

    def logits_with_hidden(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits_with_hidden(inputs, self.hidden(inputs))


class IdentitySite(nn.Module):
    """Site whose hidden IS the (flattened) input; head is identity-ish.

    Useful for hand-computed interchange tests: with an identity rotation the
    aligned coordinates are exactly the input coordinates.
    """

    def __init__(self, d: int, n_labels: int) -> None:
        super().__init__()
        self.d = d
        self.n_labels = n_labels
        self.head = nn.Linear(d, n_labels, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.reshape(inputs.shape[0], -1)

    def logits_with_hidden(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits_with_hidden(inputs, self.hidden(inputs))
