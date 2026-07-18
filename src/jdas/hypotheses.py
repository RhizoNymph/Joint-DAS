"""Boolean hypothesis library and learned-solution classifier (toy-model science).

This module provides the *measurement* tooling shared by three toy-model
experiments:

- the wrong-composition falsification baseline (``jdas run toy``, method
  ``das_wrong_and``),
- the discrete search baseline (``jdas run search``),
- the seed/basis variance study (``jdas run seed-study``).

Everything here operates on the two ground-truth boolean atoms of a toy-model
task.  For both tasks ``k_gt == 2`` and the atoms are called ``(V0, V1)``:

- ``hierarchical_equality``: ``V0 = E1 = (a == b)``, ``V1 = E2 = (c == d)``.
  True composition law: ``y = XNOR(E1, E2) = (E1 == E2)``.
- ``boolean_comp``: ``V0 = A = (x1 & x2)``, ``V1 = x3``.
  True composition law: ``y = OR(A, x3)``.

The two central objects are:

- :func:`hypothesis_library` -- a task-specific dict of named candidate binary
  variables, each a pure function of the two GT atoms, evaluated on a tensor of
  atom values.
- :func:`classify_solution` -- given the value tables of a set of *live* learned
  variables (each a function of the GT atoms over a probe set), decide whether
  the learned solution recovered the GT ``atoms``, an ``equivalent_basis``, an
  ``output_copy``, or something ``other``.

All boolean functions are named from a fixed 2-input truth-function table so a
learned variable can be labelled by its best-matching function.
"""

from __future__ import annotations

from itertools import product

import torch

# -- 2-input boolean function table -------------------------------------------

# All functions of two binary inputs (a, b) that we care to name.  Keyed by
# name -> callable(a, b) -> int in {0, 1}, with a, b long tensors.  We include
# the two projections, their negations, the two constants, and the six
# non-degenerate binary gates (AND, OR, NAND, NOR, XOR, XNOR).  ``truth_table``
# below turns any of these into its 4-bit signature over (a, b) in
# {00, 01, 10, 11}.

BOOL_FNS: dict[str, "callable"] = {
    "const0": lambda a, b: torch.zeros_like(a),
    "const1": lambda a, b: torch.ones_like(a),
    "A": lambda a, b: a,
    "B": lambda a, b: b,
    "notA": lambda a, b: 1 - a,
    "notB": lambda a, b: 1 - b,
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XOR": lambda a, b: a ^ b,
    "XNOR": lambda a, b: 1 - (a ^ b),
}

# The ten *non-constant* functions (used when reporting a live variable's best
# matching function name -- a constant is never a useful live variable label).
NONTRIVIAL_FN_NAMES: tuple[str, ...] = (
    "A",
    "B",
    "notA",
    "notB",
    "AND",
    "OR",
    "NAND",
    "NOR",
    "XOR",
    "XNOR",
)

# The four (a, b) input combinations in a fixed order.
_AB = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)


def truth_table(name: str) -> tuple[int, int, int, int]:
    """4-bit truth-table signature of ``BOOL_FNS[name]`` over (a, b).

    Order is ``(f(0,0), f(0,1), f(1,0), f(1,1))``.
    """
    fn = BOOL_FNS[name]
    a = _AB[:, 0]
    b = _AB[:, 1]
    out = fn(a, b)
    return tuple(int(x) for x in out.tolist())  # type: ignore[return-value]


def best_matching_fn(values: torch.Tensor, atoms: torch.Tensor) -> tuple[str, float]:
    """Best-matching named boolean function for a variable's value table.

    Parameters
    ----------
    values:
        ``(N,)`` long in ``{0, 1}`` -- the variable's value on each probe input.
    atoms:
        ``(N, 2)`` long -- the two GT atom values on the same probe inputs.

    Returns
    -------
    tuple[str, float]
        ``(function_name, agreement)`` over the non-constant functions, where
        ``agreement`` is the fraction of probe inputs whose value matches the
        function of the atoms.
    """
    a = atoms[:, 0]
    b = atoms[:, 1]
    best_name = "A"
    best_acc = -1.0
    for name in NONTRIVIAL_FN_NAMES:
        target = BOOL_FNS[name](a, b)
        acc = (values == target).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_name = name
    return best_name, best_acc


# -- per-task hypothesis library ----------------------------------------------


def hypothesis_library(task_name: str, atoms: torch.Tensor) -> dict[str, torch.Tensor]:
    """Named candidate binary variables computed from the GT atoms.

    ``atoms`` is ``(N, 2)`` long with columns ``(V0, V1)`` (task-specific
    meaning; see the module docstring).  Returns an ordered dict of name ->
    ``(N,)`` long candidate values.  The library is exactly the six candidates
    called out in the design for each task (15 unordered pairs of distinct
    candidates).

    - ``hierarchical_equality``: ``{E1, E2, XNOR(=y), AND, OR, NAND}``.
    - ``boolean_comp``: ``{A, x3, OR(=y), x1_only-N/A -> V0-as-A, ...}`` ->
      concretely ``{A, x3, OR(=y), notA, notX3, XOR}`` where the extra
      candidates (x1, x2 are not separately observable through the two atoms;
      the design's ``x1``/``x2`` are stand-ins for "components of A", so we use
      the two atoms and additional gates to make a 6-candidate library).
    """
    v0 = atoms[:, 0]
    v1 = atoms[:, 1]
    if task_name == "hierarchical_equality":
        return {
            "E1": v0,
            "E2": v1,
            "XNOR(=y)": 1 - (v0 ^ v1),
            "AND": v0 & v1,
            "OR": v0 | v1,
            "NAND": 1 - (v0 & v1),
        }
    if task_name == "boolean_comp":
        return {
            "A": v0,
            "x3": v1,
            "OR(=y)": v0 | v1,
            "notA": 1 - v0,
            "notx3": 1 - v1,
            "XOR": v0 ^ v1,
        }
    raise ValueError(f"unknown task {task_name!r} for hypothesis library")


# -- solution classifier ------------------------------------------------------


def _cell_purity(values: torch.Tensor, atoms: torch.Tensor) -> float:
    """Mean modal purity of ``values`` across the four (atom0, atom1) cells.

    For each of the (up to) 4 combinations of ``(atoms[:,0], atoms[:,1])`` we
    take the modal value of ``values`` in that cell and its purity (fraction of
    rows in the cell that take the modal value); we return the mean over
    non-empty cells.  Purity 1.0 means ``values`` is a deterministic function of
    the two atoms.
    """
    a = atoms[:, 0]
    b = atoms[:, 1]
    purities: list[float] = []
    for av, bv in product(range(2), repeat=2):
        sel = (a == av) & (b == bv)
        n = int(sel.sum())
        if n == 0:
            continue
        cell = values[sel]
        modal = int(torch.mode(cell).values)
        purities.append(float((cell == modal).float().mean()))
    if not purities:
        return 0.0
    return sum(purities) / len(purities)


def _symmetry_class(a: int, b: int) -> tuple[int, int]:
    """The E1<->E2-symmetric class of an atom pair (unordered pair)."""
    return (a, b) if a <= b else (b, a)


def _joint_determines_atoms(
    var_a: torch.Tensor,
    var_b: torch.Tensor,
    atoms: torch.Tensor,
) -> bool:
    """Whether ``(var_a, var_b)`` recover ``(E1, E2)`` bijectively *up to relabel*.

    The two Phase-A tasks are symmetric under the ``E1 <-> E2`` swap, so a valid
    alternative basis (e.g. ``(OR, NAND)``) can only ever determine the atoms up
    to that symmetry -- exactly why each such variable's marginal agreement with
    a single atom caps at ~0.75.  We therefore require the map ``(var_a, var_b)
    -> symmetry-class of (atom0, atom1)`` to be:

    - a *function*: each observed var-pair maps to a single symmetry class;
    - *bijective onto all classes*: every symmetry class of the atoms is
      covered, and by a distinct var-pair (so the pair carries the full
      atom-information, not just the label).

    Under the E1<->E2 symmetry there are three classes for binary atoms:
    ``{(0,0)}``, ``{(0,1),(1,0)}``, ``{(1,1)}``; ``(OR, NAND)`` hits all three
    with three distinct var-pairs and so passes, while an output copy (which
    collapses to the two label values) does not.
    """
    pair_v = (var_a * 2 + var_b).tolist()
    classes = [
        _symmetry_class(int(a), int(b))
        for a, b in zip(atoms[:, 0].tolist(), atoms[:, 1].tolist(), strict=True)
    ]
    # Each observed var-pair must map to exactly one symmetry class.
    seen: dict[int, tuple[int, int]] = {}
    for pv, cls in zip(pair_v, classes, strict=True):
        if pv in seen:
            if seen[pv] != cls:
                return False
        else:
            seen[pv] = cls
    all_classes = set(classes)
    covered = set(seen.values())
    # Cover every atom symmetry class, each by a distinct var-pair (injective).
    if covered != all_classes:
        return False
    return len(set(seen.keys())) == len(covered)


def _output_label(atoms: torch.Tensor, task_name: str) -> torch.Tensor:
    """True task label ``y`` as a function of the two atoms."""
    v0 = atoms[:, 0]
    v1 = atoms[:, 1]
    if task_name == "hierarchical_equality":
        return 1 - (v0 ^ v1)  # XNOR
    if task_name == "boolean_comp":
        return v0 | v1  # OR
    raise ValueError(f"unknown task {task_name!r}")


def classify_solution(
    live_values: list[torch.Tensor],
    atoms: torch.Tensor,
    task_name: str = "hierarchical_equality",
    *,
    purity_thresh: float = 0.9,
) -> str:
    """Classify a learned solution from its live variables' value tables.

    Parameters
    ----------
    live_values:
        List of ``(N,)`` long tensors, one per *live* learned variable, giving
        that variable's value on each of ``N`` probe inputs.
    atoms:
        ``(N, 2)`` long -- the two GT atom values on the same probe inputs.
    task_name:
        Task name (selects the output label rule for the ``output_copy`` test).
    purity_thresh:
        Purity threshold for "clean" atom / basis matches (default 0.9).

    Returns
    -------
    str
        One of ``"atoms"``, ``"equivalent_basis"``, ``"output_copy"``,
        ``"other"``.

    Classification rules (in order):

    - ``atoms``: exactly two live vars whose *individual* cell-purity vs each
      GT atom is ``>= purity_thresh`` and which between them cover both atoms
      (one matches E1, the other E2), up to relabel and swap.
    - ``equivalent_basis``: exactly two live vars whose *joint* table has cell
      purity ``>= purity_thresh`` and which jointly determine ``(E1, E2)``
      bijectively (e.g. ``(OR, NAND)``), but which are not literally the atoms.
    - ``output_copy``: some live var matches ``y`` or ``~y`` (agreement ``>=
      purity_thresh``) and the remaining live vars do not complete a basis.
    - ``other``: anything else (degenerate / >2 useful vars / partial).
    """
    # Comparisons below mix these with freshly-built CPU tensors, so normalize
    # everything to CPU (classification is cheap; inputs may arrive on CUDA).
    live_values = [v.detach().cpu() for v in live_values]
    atoms = atoms.detach().cpu()
    n_live = len(live_values)

    # -- two-variable structured solutions --------------------------------
    if n_live == 2:
        va, vb = live_values
        # atoms: each live var cleanly matches a *distinct* single atom.
        a_atom = _best_single_atom_index(va, atoms, purity_thresh)
        b_atom = _best_single_atom_index(vb, atoms, purity_thresh)
        pure_a = _cell_purity(va, atoms) >= purity_thresh
        pure_b = _cell_purity(vb, atoms) >= purity_thresh
        if (
            a_atom is not None
            and b_atom is not None
            and a_atom != b_atom
            and pure_a
            and pure_b
        ):
            return "atoms"

        # equivalent basis: joint table pure + bijectively determines atoms.
        joint_pure = _joint_pair_purity(va, vb, atoms) >= purity_thresh
        if joint_pure and _joint_determines_atoms(va, vb, atoms):
            return "equivalent_basis"

    # -- output-copy detection --------------------------------------------
    y = _output_label(atoms, task_name)
    for vals in live_values:
        agree = (vals == y).float().mean().item()
        agree = max(agree, (vals == (1 - y)).float().mean().item())
        if agree >= purity_thresh:
            # A live var copies the output; the rest must NOT complete a basis.
            others = [v for v in live_values if v is not vals]
            if len(others) == 1:
                if not (
                    _joint_pair_purity(vals, others[0], atoms) >= purity_thresh
                    and _joint_determines_atoms(vals, others[0], atoms)
                ):
                    return "output_copy"
            else:
                return "output_copy"

    return "other"


def _best_single_atom_index(
    values: torch.Tensor, atoms: torch.Tensor, thresh: float
) -> int | None:
    """Index of the single atom (0 or 1) ``values`` matches (relabel) >= thresh."""
    best_idx: int | None = None
    best_acc = thresh
    for j in range(atoms.shape[1]):
        target = atoms[:, j]
        acc = (values == target).float().mean().item()
        acc = max(acc, (values == (1 - target)).float().mean().item())
        if acc >= best_acc:
            best_acc = acc
            best_idx = j
    return best_idx


def _joint_pair_purity(
    var_a: torch.Tensor, var_b: torch.Tensor, atoms: torch.Tensor
) -> float:
    """Mean modal purity of the atom *symmetry class* given ``(var_a, var_b)``.

    For each observed ``(var_a, var_b)`` combination, how deterministically it
    pins down the E1<->E2-symmetric class of ``(atom0, atom1)`` (see
    :func:`_joint_determines_atoms` for why the symmetry class is the right
    target).  Symmetric complement of :func:`_cell_purity` (atoms -> vars
    direction); high in both directions means a bijection up to relabel.
    """
    pair_v = (var_a * 2 + var_b).long()
    cls_id = torch.tensor(
        [
            _symmetry_class(int(a), int(b))[0] * 2 + _symmetry_class(int(a), int(b))[1]
            for a, b in zip(atoms[:, 0].tolist(), atoms[:, 1].tolist(), strict=True)
        ],
        dtype=torch.long,
    )
    purities: list[float] = []
    for pv in pair_v.unique().tolist():
        sel = pair_v == pv
        cell = cls_id[sel]
        modal = int(torch.mode(cell).values)
        purities.append(float((cell == modal).float().mean()))
    if not purities:
        return 0.0
    return sum(purities) / len(purities)
