"""Tests for the toy MLP, its intervention site, and training."""

from __future__ import annotations

import pytest
import torch

from jdas.models.toy import (
    MLPSite,
    ToyMLP,
    load_or_train_toy_model,
    train_toy_model,
)
from jdas.tasks import BooleanCompositionTask, HierarchicalEqualityTask


def _make_site(input_dim: int, hidden: int = 32, n_layers: int = 3) -> MLPSite:
    torch.manual_seed(0)
    model = ToyMLP(input_dim=input_dim, hidden=hidden, n_layers=n_layers, n_labels=2)
    return MLPSite(model=model, layer_idx=1)


def test_toymlp_forward_shape() -> None:
    model = ToyMLP(input_dim=12, hidden=32, n_layers=3, n_labels=2)
    x = torch.randn(5, 12)
    assert model(x).shape == (5, 2)


def test_site_dims() -> None:
    site = _make_site(12, hidden=32)
    assert site.d == 32
    assert site.n_labels == 2


def test_site_params_frozen() -> None:
    site = _make_site(12)
    assert all(not p.requires_grad for p in site.model.parameters())


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_logits_with_hidden_matches_full(layer_idx: int) -> None:
    torch.manual_seed(1)
    model = ToyMLP(input_dim=12, hidden=32, n_layers=3, n_labels=2)
    site = MLPSite(model=model, layer_idx=layer_idx)
    x = torch.randn(7, 12)
    full = site.logits(x)
    reconstructed = site.logits_with_hidden(x, site.hidden(x))
    assert torch.equal(full, reconstructed)


def test_hidden_shape() -> None:
    site = _make_site(12, hidden=32)
    x = torch.randn(4, 12)
    assert site.hidden(x).shape == (4, 32)


def test_full_hidden_swap_yields_source_logits() -> None:
    """Substituting a source's full hidden vector reproduces the source logits."""
    torch.manual_seed(2)
    model = ToyMLP(input_dim=12, hidden=32, n_layers=3, n_labels=2)
    site = MLPSite(model=model, layer_idx=1)
    base = torch.randn(6, 12)
    source = torch.randn(6, 12)
    swapped = site.logits_with_hidden(base, site.hidden(source))
    assert torch.equal(swapped, site.logits(source))


def test_hidden_keeps_grad_graph() -> None:
    """A grad built through a substituted hidden should flow (weights frozen)."""
    site = _make_site(12, hidden=32)
    x = torch.randn(3, 12)
    h = site.hidden(x).clone().requires_grad_(True)
    out = site.logits_with_hidden(x, h).sum()
    out.backward()
    assert h.grad is not None
    assert h.grad.shape == h.shape


@pytest.mark.parametrize(
    "task_factory",
    [
        lambda: HierarchicalEqualityTask(n_emb=8),
        lambda: BooleanCompositionTask(n_emb=8, seed=0),
    ],
    ids=["heq", "boolean"],
)
def test_train_toy_model_reaches_target(task_factory) -> None:
    task = task_factory()
    device = torch.device("cpu")
    model = train_toy_model(
        task,
        device,
        steps=800,
        batch=256,
        lr=2e-3,
        seed=0,
        hidden=64,
        n_layers=3,
    )
    # verify > 99% on a fresh eval set
    gen = torch.Generator().manual_seed(999)
    inputs, labels = task.sample_inputs(4000, gen)
    acc = (model(inputs).argmax(dim=-1) == labels).float().mean().item()
    assert acc > 0.99, f"accuracy {acc}"


def test_load_or_train_caches(tmp_path) -> None:
    task = HierarchicalEqualityTask(n_emb=8)
    device = torch.device("cpu")
    cache = str(tmp_path / "ckpts")
    site1 = load_or_train_toy_model(
        task,
        site_layer=1,
        device=device,
        cache_dir=cache,
        seed=0,
        hidden=64,
        n_layers=3,
        steps=800,
        batch=256,
        lr=2e-3,
    )
    assert isinstance(site1, MLPSite)
    # second call loads from cache -> identical weights
    site2 = load_or_train_toy_model(
        task,
        site_layer=2,
        device=device,
        cache_dir=cache,
        seed=0,
        hidden=64,
        n_layers=3,
    )
    for p1, p2 in zip(site1.model.parameters(), site2.model.parameters()):
        assert torch.equal(p1, p2)
    assert site2.layer_idx == 2
