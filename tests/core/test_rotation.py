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


# -- width caps (max_width) ----------------------------------------------------


def test_default_softplus_widths_unchanged_golden() -> None:
    """Default (max_width=None) path matches the historical softplus behavior.

    Golden values captured from the pre-change implementation: init_width=2.0
    gives every soft width == 2.0, total == 6.0, hard_widths == [2, 2, 2].
    """
    layout = SubspaceLayout(16, 3, init_width=2.0)
    assert layout.max_width is None
    torch.testing.assert_close(
        layout.widths(), torch.tensor([2.0, 2.0, 2.0]), atol=1e-5, rtol=0
    )
    assert float(layout.total_aligned_dims()) == pytest.approx(6.0, abs=1e-5)
    assert layout.hard_widths().tolist() == [2, 2, 2]


def test_bounded_init_matches_init_width() -> None:
    """Bounded mode initializes each width ~= init_width."""
    layout = SubspaceLayout(64, 4, init_width=8.0, max_width=32.0)
    torch.testing.assert_close(
        layout.widths(),
        torch.full((4,), 8.0),
        atol=1e-4,
        rtol=0,
    )


def test_bounded_width_cap_respected_under_pressure() -> None:
    """Even after aggressive optimization rewarding width, w_i < max_width."""
    torch.manual_seed(0)
    d, k, cap = 256, 3, 32.0
    layout = SubspaceLayout(d, k, init_width=8.0, max_width=cap)
    opt = torch.optim.SGD(layout.parameters(), lr=10.0)
    # Loss that rewards larger widths -> pushes raw_widths up hard.
    for _ in range(200):
        opt.zero_grad()
        loss = -layout.widths().sum()
        loss.backward()
        opt.step()
    w = layout.widths()
    # Bounded parameterization guarantees w <= max_width for all raw values;
    # in float32 sigmoid saturates to exactly 1.0 under extreme pressure, so the
    # invariant is "never exceeds the cap" (the hard cap is respected).
    assert (w <= cap).all(), w
    # And the hard integer widths never exceed the cap either.
    assert int(layout.hard_widths().max()) <= int(cap)


def test_bounded_requires_init_below_cap() -> None:
    """init_width must be strictly below max_width."""
    with pytest.raises(Exception):
        SubspaceLayout(16, 2, init_width=32.0, max_width=32.0)
    with pytest.raises(Exception):
        SubspaceLayout(16, 2, init_width=40.0, max_width=32.0)
