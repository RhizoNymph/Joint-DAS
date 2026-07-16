"""Tests for eval metrics: recovery and IIA sanity."""

from __future__ import annotations

import torch

from jdas.causal_model import FixedCausalModel
from jdas.eval import iia, recovery
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.types import InterventionBatch

from .fakes import XorTask


def _bits(inputs: torch.Tensor, emb: int = 3) -> torch.Tensor:
    x = inputs.reshape(inputs.shape[0], 2, emb)
    return (x.mean(dim=-1) > 0).long()


def test_recovery_perfect_for_gt_model() -> None:
    """A FixedCausalModel equal to GT recovers with score ~1.0."""
    task = XorTask(emb=3)
    model = FixedCausalModel(
        lambda x: _bits(x, task.emb),
        task.label_from_variables,
        k=2,
        v=2,
        n_labels=2,
    )
    gen = torch.Generator().manual_seed(0)
    rec = recovery(model, task, n_samples=256, generator=gen)
    assert rec.best_score > 0.99
    # best assignment should map gt vars to distinct learned vars
    assert len(set(rec.best_assignment)) == len(rec.best_assignment)


def test_recovery_near_chance_for_random_model() -> None:
    """A model whose variables are random-ish yields near-chance recovery."""
    task = XorTask(emb=3)

    def random_vars(inputs: torch.Tensor) -> torch.Tensor:
        # deterministic pseudo-random bits independent of gt structure
        h = (inputs.reshape(inputs.shape[0], -1).sum(-1) * 1000).long()
        b0 = (h % 2)
        b1 = ((h // 3) % 2)
        return torch.stack([b0, b1], dim=1)

    model = FixedCausalModel(random_vars, task.label_from_variables, k=2, v=2, n_labels=2)
    gen = torch.Generator().manual_seed(1)
    rec = recovery(model, task, n_samples=512, generator=gen)
    # Best relabel agreement of an unrelated bit should be near 0.5.
    assert rec.best_score < 0.75


def test_iia_sanity_network_is_causal_model() -> None:
    """On a rigged site where N literally computes H's XOR, IIA ~ 1.0.

    Build a site whose hidden encodes the two bits in disjoint dims and whose
    head computes XOR of those dims; align rotation=identity and layout to the
    two bit-dims.  Then interchange swaps bits exactly as H does.
    """
    task = XorTask(emb=1)  # each block is a single scalar; bit = (x>0)
    d = 2
    # Site: hidden[:, i] = sign-ish of block i; head computes XOR via a table.
    from torch import nn

    class RiggedSite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d = d
            self.n_labels = 2

        def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
            # hidden dim i = the raw scalar of block i (emb=1)
            return inputs.reshape(inputs.shape[0], 2)

        def logits_with_hidden(self, inputs, hidden):
            b0 = (hidden[:, 0] > 0).long()
            b1 = (hidden[:, 1] > 0).long()
            y = b0 ^ b1
            return torch.nn.functional.one_hot(y, 2).float() * 10.0

        def logits(self, inputs):
            return self.logits_with_hidden(inputs, self.hidden(inputs))

    site = RiggedSite()
    model = FixedCausalModel(
        lambda x: _bits(x, task.emb), task.label_from_variables, k=2, v=2, n_labels=2
    )
    rot = OrthogonalRotation(d)
    rot.set_matrix(torch.eye(d))
    layout = SubspaceLayout(d, 2, init_width=1.0)
    with torch.no_grad():
        raw = torch.log(torch.expm1(torch.tensor(1.0)))
        layout.raw_widths.fill_(float(raw))

    gen = torch.Generator().manual_seed(3)
    scores = iia(
        site, rot, layout, model, task,
        n_batches=4, batch_size=64, n_sources=2, generator=gen, swap_sizes=(1, 2),
    )
    assert scores[1] > 0.95
    assert scores[2] > 0.95
