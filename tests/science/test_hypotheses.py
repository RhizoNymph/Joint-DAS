"""Tests for the boolean hypothesis library and solution classifier.

The classifier is the load-bearing measurement of the seed/basis study, so it
is exercised on hand-built synthetic variable assignments covering every class:
``atoms``, ``equivalent_basis`` (e.g. (OR, NAND)), ``output_copy``, and
degenerate / ``other`` cases.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from jdas.hypotheses import (
    BOOL_FNS,
    NONTRIVIAL_FN_NAMES,
    best_matching_fn,
    classify_solution,
    hypothesis_library,
    truth_table,
)


def _all_atoms(reps: int = 300) -> torch.Tensor:
    """A balanced probe over the four (E1, E2) combinations, ``reps`` each."""
    base = torch.tensor(
        list(itertools.product([0, 1], repeat=2)), dtype=torch.long
    )  # (4, 2)
    return base.repeat(reps, 1)  # (4*reps, 2)


# -- truth table / boolean function names -------------------------------------


def test_truth_tables() -> None:
    assert truth_table("AND") == (0, 0, 0, 1)
    assert truth_table("OR") == (0, 1, 1, 1)
    assert truth_table("NAND") == (1, 1, 1, 0)
    assert truth_table("NOR") == (1, 0, 0, 0)
    assert truth_table("XOR") == (0, 1, 1, 0)
    assert truth_table("XNOR") == (1, 0, 0, 1)
    assert truth_table("A") == (0, 0, 1, 1)
    assert truth_table("B") == (0, 1, 0, 1)


def test_best_matching_fn_identifies_or() -> None:
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    vals = a | b
    name, acc = best_matching_fn(vals, atoms)
    assert name == "OR"
    assert acc == pytest.approx(1.0)


def test_best_matching_fn_identifies_nand() -> None:
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    vals = 1 - (a & b)
    name, acc = best_matching_fn(vals, atoms)
    assert name == "NAND"
    assert acc == pytest.approx(1.0)


def test_best_matching_fn_only_nontrivial() -> None:
    """A constant variable still returns *some* non-constant best match."""
    atoms = _all_atoms()
    vals = torch.zeros(atoms.shape[0], dtype=torch.long)
    name, _ = best_matching_fn(vals, atoms)
    assert name in NONTRIVIAL_FN_NAMES


# -- hypothesis library -------------------------------------------------------


def test_hypothesis_library_heq() -> None:
    atoms = _all_atoms()
    lib = hypothesis_library("hierarchical_equality", atoms)
    assert set(lib) == {"E1", "E2", "XNOR(=y)", "AND", "OR", "NAND"}
    a, b = atoms[:, 0], atoms[:, 1]
    assert torch.equal(lib["E1"], a)
    assert torch.equal(lib["E2"], b)
    assert torch.equal(lib["XNOR(=y)"], 1 - (a ^ b))
    assert torch.equal(lib["NAND"], 1 - (a & b))


def test_hypothesis_library_boolean() -> None:
    atoms = _all_atoms()
    lib = hypothesis_library("boolean_comp", atoms)
    assert set(lib) == {"A", "x3", "OR(=y)", "notA", "notx3", "XOR"}
    a, b = atoms[:, 0], atoms[:, 1]
    assert torch.equal(lib["OR(=y)"], a | b)
    assert torch.equal(lib["XOR"], a ^ b)


def test_hypothesis_library_unknown_task() -> None:
    with pytest.raises(ValueError):
        hypothesis_library("nope", _all_atoms())


# -- classifier: atoms --------------------------------------------------------


def test_classify_atoms_exact() -> None:
    atoms = _all_atoms()
    live = [atoms[:, 0], atoms[:, 1]]  # Z0 = E1, Z1 = E2
    assert classify_solution(live, atoms, "hierarchical_equality") == "atoms"


def test_classify_atoms_relabeled_and_swapped() -> None:
    """atoms up to value-relabel (negation) and variable swap."""
    atoms = _all_atoms()
    live = [atoms[:, 1], 1 - atoms[:, 0]]  # Z0 = E2, Z1 = notE1
    assert classify_solution(live, atoms, "hierarchical_equality") == "atoms"


def test_classify_atoms_noisy_but_above_threshold() -> None:
    """A few percent noise still classifies as atoms (purity >= 0.9)."""
    torch.manual_seed(0)
    atoms = _all_atoms(reps=500)
    z0 = atoms[:, 0].clone()
    z1 = atoms[:, 1].clone()
    # flip 5% of z0.
    flip = torch.rand(z0.shape[0]) < 0.05
    z0 = torch.where(flip, 1 - z0, z0)
    assert classify_solution([z0, z1], atoms, "hierarchical_equality") == "atoms"


# -- classifier: equivalent basis --------------------------------------------


def test_classify_equivalent_basis_or_nand() -> None:
    """(OR, NAND) is a valid alternative basis, not the literal atoms."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    z0 = a | b  # OR
    z1 = 1 - (a & b)  # NAND
    # (OR, NAND) jointly determine (E1, E2) up to the E1<->E2 symmetry.
    assert classify_solution([z0, z1], atoms, "hierarchical_equality") == "equivalent_basis"


def test_classify_equivalent_basis_and_or() -> None:
    """(AND, OR) also jointly determine the atoms up to symmetry."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    z0 = a & b  # AND
    z1 = a | b  # OR
    assert classify_solution([z0, z1], atoms, "hierarchical_equality") == "equivalent_basis"


def test_equivalent_basis_not_confused_with_atoms() -> None:
    """(OR, NAND) must NOT be classified as atoms (single-atom purity fails)."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    assert classify_solution([a | b, 1 - (a & b)], atoms) != "atoms"


# -- classifier: output copy --------------------------------------------------


def test_classify_output_copy_single_live() -> None:
    """One live var == y (XNOR) and nothing to complete a basis."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    y = 1 - (a ^ b)  # XNOR == label for heq
    assert classify_solution([y], atoms, "hierarchical_equality") == "output_copy"


def test_classify_output_copy_with_junk_partner() -> None:
    """y plus a constant partner: still output copy (no basis)."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    y = 1 - (a ^ b)
    junk = torch.zeros_like(y)
    assert classify_solution([y, junk], atoms, "hierarchical_equality") == "output_copy"


def test_classify_output_copy_boolean_comp() -> None:
    """For boolean_comp the label is OR; a lone OR var is an output copy."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    y = a | b  # OR == label for boolean_comp
    assert classify_solution([y], atoms, "boolean_comp") == "output_copy"


# -- classifier: degenerate / other ------------------------------------------


def test_classify_degenerate_constants() -> None:
    """All-constant live vars determine nothing -> other."""
    atoms = _all_atoms()
    z0 = torch.zeros(atoms.shape[0], dtype=torch.long)
    z1 = torch.ones(atoms.shape[0], dtype=torch.long)
    assert classify_solution([z0, z1], atoms, "hierarchical_equality") == "other"


def test_classify_single_atom_not_basis() -> None:
    """A single live var that is one atom (not the label) is not a full basis."""
    atoms = _all_atoms()
    # E1 alone: not output copy (E1 != XNOR), not two-var structured -> other.
    assert classify_solution([atoms[:, 0]], atoms, "hierarchical_equality") == "other"


def test_classify_empty_live() -> None:
    atoms = _all_atoms()
    assert classify_solution([], atoms, "hierarchical_equality") == "other"


def test_classify_three_live_vars_other() -> None:
    """More than two live vars (none an output copy) -> other."""
    atoms = _all_atoms()
    a, b = atoms[:, 0], atoms[:, 1]
    live = [a, b, a & b]  # E1, E2, AND -- three live, no output copy of y
    assert classify_solution(live, atoms, "hierarchical_equality") == "other"


def test_classify_boolean_comp_atoms() -> None:
    atoms = _all_atoms()
    assert classify_solution([atoms[:, 0], atoms[:, 1]], atoms, "boolean_comp") == "atoms"
