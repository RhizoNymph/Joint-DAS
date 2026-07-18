"""Tests for the wrong-composition (das_wrong_and) falsification baseline.

Covers the k=2 FixedCausalModel with the true GT variables + a wrong law, and
the analytic agreement ceiling (fraction of interventions where wrong-law and
true-law counterfactual labels coincide, per swap size).
"""

from __future__ import annotations

import argparse

import pytest
import torch

from jdas.cli.runners import (
    _load_task_toy as _load_task,
    _wrong_and_agreement_ceiling_toy as _wrong_and_agreement_ceiling,
    _wrong_and_fixed_model_toy as _wrong_and_fixed_model,
    _wrong_law_label_fn,
)


def _args(task: str) -> argparse.Namespace:
    return argparse.Namespace(task=task, seed=0, n_sources=2, v=2)


def test_wrong_law_heq_is_and() -> None:
    """hierarchical_equality wrong law is AND (truth is XNOR)."""
    fn = _wrong_law_label_fn("hierarchical_equality")
    vals = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert fn(vals).tolist() == [0, 0, 0, 1]  # AND


def test_wrong_law_boolean_is_xor() -> None:
    """boolean_comp wrong law is XOR (truth is OR)."""
    fn = _wrong_law_label_fn("boolean_comp")
    vals = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert fn(vals).tolist() == [0, 1, 1, 0]  # XOR


def test_wrong_and_model_is_k2_true_vars() -> None:
    """The model has k=2 and uses the task's TRUE ground-truth variables."""
    task = _load_task("hierarchical_equality")
    model = _wrong_and_fixed_model(task, _args("hierarchical_equality"))
    assert model.k_max == 2
    assert model.v == 2
    gen = torch.Generator().manual_seed(0)
    batch = task.sample_batch(64, 2, 2, gen)
    # variable_values must equal the task's own gt_variables (true atoms).
    assert torch.equal(model.variable_values(batch.base_inputs), task.gt_variables(batch.base_inputs))


def test_wrong_and_model_wrong_clean_label() -> None:
    """The clean predicted label follows the WRONG law, not the task label."""
    task = _load_task("hierarchical_equality")
    model = _wrong_and_fixed_model(task, _args("hierarchical_equality"))
    gen = torch.Generator().manual_seed(1)
    inputs, true_labels = task.sample_inputs(2000, gen)
    pred = model.predict(inputs).argmax(-1)
    atoms = task.gt_variables(inputs)
    wrong = (atoms[:, 0] & atoms[:, 1]).long()  # AND
    assert torch.equal(pred, wrong)
    # And it genuinely disagrees with the true label on a nontrivial fraction.
    disagree = (pred != true_labels).float().mean().item()
    assert disagree > 0.1


def test_agreement_ceiling_heq_near_three_quarters() -> None:
    """AND vs XNOR agree on 3/4 of the four atom combinations (~0.75)."""
    task = _load_task("hierarchical_equality")
    ceil = _wrong_and_agreement_ceiling(task, _args("hierarchical_equality"), n_samples=20000)
    assert set(ceil) == {"1", "2"}
    for s in ("1", "2"):
        assert ceil[s] == pytest.approx(0.75, abs=0.03)


def test_agreement_ceiling_boolean() -> None:
    """XOR vs OR: agree everywhere except (A,x3)=(1,1); weighted ceiling ~0.87."""
    task = _load_task("boolean_comp")
    ceil = _wrong_and_agreement_ceiling(task, _args("boolean_comp"), n_samples=20000)
    assert set(ceil) == {"1", "2"}
    for s in ("1", "2"):
        assert 0.83 < ceil[s] < 0.91


def test_agreement_ceiling_is_below_one() -> None:
    """The ceiling must be strictly below 1: the falsification has teeth."""
    for task_name in ("hierarchical_equality", "boolean_comp"):
        task = _load_task(task_name)
        ceil = _wrong_and_agreement_ceiling(task, _args(task_name), n_samples=8000)
        assert all(0.5 < v < 0.95 for v in ceil.values())
