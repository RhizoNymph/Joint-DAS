"""Tests for OrthogonalRotation and SubspaceLayout."""

from __future__ import annotations

import pytest
import torch

from jdas.rotation import OrthogonalRotation, SubspaceLayout


def test_rotation_orthogonal_after_optimization() -> None:
    """Q stays orthogonal (Q Q^T = I) after several optimization steps."""
    torch.manual_seed(0)
    d = 8
    rot = OrthogonalRotation(d)
    opt = torch.optim.SGD(rot.parameters(), lr=0.1)
    target = torch.randn(4, d)
    for _ in range(20):
        opt.zero_grad()
        out = rot.rotate(torch.randn(4, d))
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    q = rot.matrix
    eye = torch.eye(d)
    assert torch.allclose(q @ q.T, eye, atol=1e-4)
    assert torch.allclose(q.T @ q, eye, atol=1e-4)


def test_rotate_unrotate_inverse() -> None:
    """unrotate(rotate(h)) == h for orthogonal Q."""
    torch.manual_seed(1)
    d = 6
    rot = OrthogonalRotation(d)
    h = torch.randn(5, d)
    r = rot.rotate(h)
    back = rot.unrotate(r)
    assert torch.allclose(back, h, atol=1e-5)


def test_rotate_orientation() -> None:
    """rotate(h) equals h @ Q^T (row p uses row p of Q)."""
    torch.manual_seed(2)
    d = 4
    rot = OrthogonalRotation(d)
    h = torch.randn(3, d)
    expected = h @ rot.matrix.T
    assert torch.allclose(rot.rotate(h), expected, atol=1e-6)


def test_frozen_rotation_no_grad() -> None:
    """A frozen rotation has no trainable parameters."""
    rot = OrthogonalRotation(5, freeze=True)
    assert all(not p.requires_grad for p in rot.parameters())


def test_layout_masks_disjoint() -> None:
    """Hard masks form a disjoint partition (no coordinate in two blocks)."""
    torch.manual_seed(3)
    d, k = 12, 3
    layout = SubspaceLayout(d, k, init_width=3.0)
    hard = layout.hard_masks()
    counts = hard.long().sum(dim=0)  # per-coordinate count over blocks
    assert (counts <= 1).all()


def test_layout_soft_masks_columns_le_one() -> None:
    """Soft masks: each coordinate's total membership is <= ~1 (disjoint)."""
    torch.manual_seed(4)
    d, k = 10, 3
    layout = SubspaceLayout(d, k, init_width=2.0)
    layout.set_temperature(0.05)
    soft = layout.soft_masks()
    col_sum = soft.sum(dim=0)
    assert (col_sum <= 1.0 + 1e-3).all()


def test_layout_hard_soft_consistency_low_temp() -> None:
    """At low temperature soft masks approach the hard boolean masks."""
    torch.manual_seed(5)
    d, k = 12, 3
    layout = SubspaceLayout(d, k, init_width=4.0)
    layout.set_temperature(0.01)
    soft = layout.soft_masks()
    hard = layout.hard_masks().float()
    assert torch.allclose(soft, hard, atol=0.05)


def test_layout_temperature_clamp() -> None:
    """Temperature is clamped to [min_temp, max_temp]."""
    layout = SubspaceLayout(8, 2, min_temp=0.1, max_temp=2.0)
    layout.set_temperature(100.0)
    assert layout.temperature == pytest.approx(2.0)
    layout.set_temperature(0.0)
    assert layout.temperature == pytest.approx(0.1)


def test_layout_total_aligned_dims_matches_boundary() -> None:
    """total_aligned_dims equals the last cumulative boundary c_{k_max}."""
    layout = SubspaceLayout(10, 3, init_width=2.0)
    total = layout.total_aligned_dims()
    assert torch.allclose(total, layout.boundaries()[-1])
    assert 0.0 <= float(total) <= 10.0


def test_layout_boundaries_clamped_to_d() -> None:
    """Cumulative boundaries never exceed d even for huge widths."""
    layout = SubspaceLayout(4, 5, init_width=1.0)
    with torch.no_grad():
        layout.raw_widths.fill_(10.0)  # huge softplus widths
    b = layout.boundaries()
    assert float(b[-1]) <= 4.0 + 1e-4
    assert (b <= 4.0 + 1e-4).all()


def test_layout_gradient_flows_to_widths() -> None:
    """Soft masks are differentiable w.r.t. the width parameters."""
    layout = SubspaceLayout(8, 2, init_width=2.0)
    layout.set_temperature(0.5)
    soft = layout.soft_masks()
    soft.sum().backward()
    assert layout.raw_widths.grad is not None
    assert layout.raw_widths.grad.abs().sum() > 0
