"""Interchange correctness on hand-computed examples with identity rotation."""

from __future__ import annotations

import torch

from jdas.intervention import interchange
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.types import InterventionBatch

from .fakes import IdentitySite


def _identity_rotation(d: int) -> OrthogonalRotation:
    """A rotation whose matrix is set to the identity."""
    rot = OrthogonalRotation(d)
    rot.set_matrix(torch.eye(d))
    return rot


def test_interchange_swaps_exact_dims_2d() -> None:
    """With identity rotation and 2 blocks of width 1, swapping var 0 copies
    exactly dim 0 from the source, leaving dim 1 from base."""
    d = 2
    site = IdentitySite(d, n_labels=2)
    with torch.no_grad():
        site.head.weight.copy_(torch.eye(d))  # logits == hidden
    rot = _identity_rotation(d)
    layout = SubspaceLayout(d, 2, init_width=1.0)
    # Force widths so block0 = dim0, block1 = dim1.
    with torch.no_grad():
        import torch.nn.functional as F

        # softplus(raw) = 1  => raw = inv_softplus(1)
        raw = torch.log(torch.expm1(torch.tensor(1.0)))
        layout.raw_widths.fill_(float(raw))

    base = torch.tensor([[10.0, 20.0]])  # (1, 2)
    source = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
    assign = torch.tensor([[0, -1]])  # swap var0 (dim0) from source0
    batch = InterventionBatch(
        base_inputs=base,
        source_inputs=source,
        source_assignment=assign,
        base_labels=torch.zeros(1, dtype=torch.long),
        source_labels=torch.zeros(1, 1, dtype=torch.long),
    )
    out = interchange(site, rot, layout, batch, hard=True)
    # dim0 <- source (1.0), dim1 stays base (20.0)
    assert torch.allclose(out, torch.tensor([[1.0, 20.0]]), atol=1e-5)


def test_interchange_no_swap_is_identity() -> None:
    """All-(-1) assignment reproduces the base output exactly."""
    d = 4
    site = IdentitySite(d, n_labels=d)
    with torch.no_grad():
        site.head.weight.copy_(torch.eye(d))
    rot = _identity_rotation(d)
    layout = SubspaceLayout(d, 2, init_width=2.0)
    base = torch.randn(3, d)
    source = torch.randn(3, 2, d)
    assign = torch.full((3, 2), -1, dtype=torch.long)
    batch = InterventionBatch(
        base_inputs=base,
        source_inputs=source,
        source_assignment=assign,
        base_labels=torch.zeros(3, dtype=torch.long),
        source_labels=torch.zeros(3, 2, dtype=torch.long),
    )
    out = interchange(site, rot, layout, batch, hard=True)
    assert torch.allclose(out, base, atol=1e-5)


def test_interchange_two_var_two_source() -> None:
    """Two blocks each width 2 over d=4; swap var0 from src0, var1 from src1."""
    d = 4
    site = IdentitySite(d, n_labels=d)
    with torch.no_grad():
        site.head.weight.copy_(torch.eye(d))
    rot = _identity_rotation(d)
    layout = SubspaceLayout(d, 2, init_width=2.0)
    with torch.no_grad():
        raw = torch.log(torch.expm1(torch.tensor(2.0)))
        layout.raw_widths.fill_(float(raw))  # each block width 2

    base = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    source = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]])  # (1,2,4)
    assign = torch.tensor([[0, 1]])  # var0(dims0-1)<-src0; var1(dims2-3)<-src1
    batch = InterventionBatch(
        base_inputs=base,
        source_inputs=source,
        source_assignment=assign,
        base_labels=torch.zeros(1, dtype=torch.long),
        source_labels=torch.zeros(1, 2, dtype=torch.long),
    )
    out = interchange(site, rot, layout, batch, hard=True)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 2.0, 2.0]]), atol=1e-5)


def test_interchange_gradient_flows_to_rotation() -> None:
    """Gradients reach the rotation params through the interchange (soft)."""
    d = 6
    site = IdentitySite(d, n_labels=2)
    rot = OrthogonalRotation(d)
    layout = SubspaceLayout(d, 2, init_width=2.0)
    layout.set_temperature(0.5)
    base = torch.randn(4, d)
    source = torch.randn(4, 2, d)
    assign = torch.tensor([[0, -1]] * 4)
    batch = InterventionBatch(
        base_inputs=base,
        source_inputs=source,
        source_assignment=assign,
        base_labels=torch.zeros(4, dtype=torch.long),
        source_labels=torch.zeros(4, 2, dtype=torch.long),
    )
    out = interchange(site, rot, layout, batch, hard=False)
    out.sum().backward()
    grads = [p.grad for p in rot.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert layout.raw_widths.grad is not None
