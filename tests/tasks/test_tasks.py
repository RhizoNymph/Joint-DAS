"""Tests for the Phase-A synthetic tasks."""

from __future__ import annotations

import pytest
import torch

from jdas.tasks import BooleanCompositionTask, HierarchicalEqualityTask
from jdas.types import InterventionBatch, Task


def _get_task(request: pytest.FixtureRequest, name: str) -> Task:
    return request.getfixturevalue(name)


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_protocol_conformance(request: pytest.FixtureRequest, name: str) -> None:
    task = _get_task(request, name)
    assert isinstance(task, Task)
    assert task.n_labels == 2
    assert task.k_gt == 2


def test_heq_label_balance(heq) -> None:
    """Hierarchical equality is constructed to be label-balanced (~50/50)."""
    gen = torch.Generator().manual_seed(0)
    _, labels = heq.sample_inputs(20000, gen)
    frac = labels.float().mean().item()
    assert 0.4 < frac < 0.6, f"labels not balanced: {frac}"


def test_bool_label_distribution(boolean) -> None:
    """Boolean composition with uniform bits has P(y=1) = 1 - 0.75*0.5 = 0.625.

    (Not 50/50: the task's natural label distribution under uniform input bits.)
    """
    gen = torch.Generator().manual_seed(0)
    _, labels = boolean.sample_inputs(20000, gen)
    frac = labels.float().mean().item()
    assert 0.60 < frac < 0.65, f"unexpected label distribution: {frac}"
    # both classes are well represented
    assert frac > 0.3 and frac < 0.7


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_gt_label_fn_reproduces_labels(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(3)
    inputs, labels = task.sample_inputs(2000, gen)
    vars_ = task.gt_variables(inputs)
    assert torch.equal(task.gt_label_fn(vars_), labels)


def test_heq_gt_variables_construction() -> None:
    task = HierarchicalEqualityTask(n_emb=8)
    gen = torch.Generator().manual_seed(11)
    inputs, _ = task.sample_inputs(1000, gen)
    a, b, c, d = inputs.view(-1, 4, 8).unbind(dim=1)
    e1 = (a == b).all(dim=-1).long()
    e2 = (c == d).all(dim=-1).long()
    vars_ = task.gt_variables(inputs)
    assert torch.equal(vars_[:, 0], e1)
    assert torch.equal(vars_[:, 1], e2)
    # both equality events should actually occur (balanced construction)
    assert e1.float().mean().item() > 0.3
    assert e2.float().mean().item() > 0.3


def test_bool_gt_variables_construction() -> None:
    task = BooleanCompositionTask(n_emb=8, seed=5)
    gen = torch.Generator().manual_seed(13)
    inputs, labels = task.sample_inputs(2000, gen)
    bits = task._bits(inputs)
    x1_and_x2 = (bits[:, 0] & bits[:, 1]).long()
    x3 = bits[:, 2].long()
    vars_ = task.gt_variables(inputs)
    assert torch.equal(vars_[:, 0], x1_and_x2)
    assert torch.equal(vars_[:, 1], x3)
    # y == (x1 & x2) | x3
    expected = (bits[:, 0] & bits[:, 1]) | bits[:, 2]
    assert torch.equal(labels, expected.long())


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_embeddings_on_unit_sphere(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(9)
    inputs, _ = task.sample_inputs(500, gen)
    n_emb = task.n_emb
    chunks = inputs.view(inputs.shape[0], -1, n_emb)
    norms = chunks.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_sample_batch_shapes(request: pytest.FixtureRequest, name: str) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(2)
    B, m, k = 64, 3, 4
    batch = task.sample_batch(B, m, k, gen)
    assert isinstance(batch, InterventionBatch)
    assert batch.base_inputs.shape == (B, task.input_dim)
    assert batch.source_inputs.shape == (B, m, task.input_dim)
    assert batch.source_assignment.shape == (B, k)
    assert batch.base_labels.shape == (B,)
    assert batch.source_labels.shape == (B, m)
    assert batch.source_assignment.dtype == torch.long
    # entries within [-1, m)
    assert batch.source_assignment.min().item() >= -1
    assert batch.source_assignment.max().item() < m


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_source_assignment_iset_sizes(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(4)
    B, m, k = 4000, 3, 4
    batch = task.sample_batch(B, m, k, gen)
    n_swapped = (batch.source_assignment >= 0).sum(dim=1)
    # |I| is always 1 or 2
    assert set(n_swapped.unique().tolist()).issubset({1, 2})
    assert (n_swapped == 1).any()
    assert (n_swapped == 2).any()
    # roughly balanced 50/50
    frac_two = (n_swapped == 2).float().mean().item()
    assert 0.4 < frac_two < 0.6


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_source_assignment_only_first_k_gt(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(6)
    B, m, k = 2000, 3, 4
    batch = task.sample_batch(B, m, k, gen)
    # only the first k_gt slots may ever be swapped
    swapped_beyond = batch.source_assignment[:, task.k_gt :]
    assert (swapped_beyond == -1).all()


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_source_assignment_distinct_for_iset_two(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(8)
    B, m, k = 4000, 3, 4
    batch = task.sample_batch(B, m, k, gen)
    sa = batch.source_assignment
    two = (sa >= 0).sum(dim=1) == 2
    rows = sa[two]
    for row in rows:
        srcs = row[row >= 0]
        assert srcs.numel() == 2
        assert srcs[0].item() != srcs[1].item()


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_counterfactual_semantics_match_bruteforce(
    request: pytest.FixtureRequest, name: str
) -> None:
    """gt_label_fn on base-vars-with-source-swaps matches a brute-force compute."""
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(21)
    B, m, k = 3000, 3, 4
    batch = task.sample_batch(B, m, k, gen)

    base_vars = task.gt_variables(batch.base_inputs)  # (B, k_gt)
    source_vars = torch.stack(
        [task.gt_variables(batch.source_inputs[:, j]) for j in range(m)], dim=1
    )  # (B, m, k_gt)

    # Build counterfactual variable assignment by explicit indexing.
    cf_vars = base_vars.clone()
    sa = batch.source_assignment  # (B, k)
    for b in range(B):
        for i in range(task.k_gt):
            j = int(sa[b, i].item())
            if j >= 0:
                cf_vars[b, i] = source_vars[b, j, i]
    cf_labels = task.gt_label_fn(cf_vars)

    # Vectorized reference implementation.
    ref_vars = base_vars.clone()
    for i in range(task.k_gt):
        j = sa[:, i]  # (B,)
        mask = j >= 0
        jj = j.clamp_min(0)
        picked = source_vars[torch.arange(B), jj, i]
        ref_vars[mask, i] = picked[mask]
    ref_labels = task.gt_label_fn(ref_vars)

    assert torch.equal(cf_labels, ref_labels)


@pytest.mark.parametrize("name", ["heq", "boolean"])
def test_base_and_source_labels_correct(
    request: pytest.FixtureRequest, name: str
) -> None:
    task = _get_task(request, name)
    gen = torch.Generator().manual_seed(31)
    B, m, k = 500, 2, 4
    batch = task.sample_batch(B, m, k, gen)
    assert torch.equal(
        batch.base_labels, task.gt_label_fn(task.gt_variables(batch.base_inputs))
    )
    for j in range(m):
        assert torch.equal(
            batch.source_labels[:, j],
            task.gt_label_fn(task.gt_variables(batch.source_inputs[:, j])),
        )


def test_heq_distinct_symbols_are_distinct() -> None:
    """When a pair is not equal-by-construction, the two vectors differ."""
    task = HierarchicalEqualityTask(n_emb=8)
    gen = torch.Generator().manual_seed(41)
    inputs, _ = task.sample_inputs(2000, gen)
    a, b, c, d = inputs.view(-1, 4, 8).unbind(dim=1)
    # where a != b as vectors, gt var should be 0; check consistency
    vars_ = task.gt_variables(inputs)
    ab_eq_vec = (a == b).all(dim=-1)
    assert torch.equal(ab_eq_vec.long(), vars_[:, 0])


def test_kmax_smaller_than_kgt() -> None:
    """k_max < k_gt only swaps within available slots."""
    task = HierarchicalEqualityTask(n_emb=8)
    gen = torch.Generator().manual_seed(51)
    B, m, k = 1000, 3, 1
    batch = task.sample_batch(B, m, k, gen)
    assert batch.source_assignment.shape == (B, 1)
    n_swapped = (batch.source_assignment >= 0).sum(dim=1)
    # with only 1 slot, |I| can only be 1
    assert set(n_swapped.unique().tolist()).issubset({1})
