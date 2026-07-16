"""Evaluation metrics for Joint-DAS.

All metrics run under :func:`torch.no_grad`, use *hard* discretization (hard
subspace masks + hard/argmax variables), and are deterministic given a
:class:`torch.Generator`.

- :func:`iia` -- interchange intervention accuracy, reported per swap size
  (``|I|=1`` and ``|I|=2`` with distinct sources).
- :func:`recovery` -- agreement matrix between learned and ground-truth
  variables, with the best value-relabeling and best variable assignment.
- :func:`effective_k` -- number of "live" variables (non-degenerate mask that
  actually changes N's output when swapped).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import torch

from .intervention import interchange
from .rotation import OrthogonalRotation, SubspaceLayout
from .types import InterventionBatch, InterventionSite, Task


class EvalError(ValueError):
    """Raised for invalid evaluation configuration."""


def _build_assignment(
    b: int,
    k_max: int,
    n_sources: int,
    swap_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Random ``source_assignment`` ``(b, k_max)`` swapping exactly ``swap_size``
    distinct variables, each from a *distinct* source (requires
    ``n_sources >= swap_size``).
    """
    if swap_size > k_max:
        raise EvalError(f"swap_size {swap_size} > k_max {k_max}")
    if swap_size > n_sources:
        raise EvalError(
            f"swap_size {swap_size} needs >= {swap_size} sources, got {n_sources}"
        )
    assign = torch.full((b, k_max), -1, dtype=torch.long, device=device)
    for row in range(b):
        var_perm = torch.randperm(k_max, generator=generator, device=device)
        vars_to_swap = var_perm[:swap_size]
        src_perm = torch.randperm(n_sources, generator=generator, device=device)
        srcs = src_perm[:swap_size]
        assign[row, vars_to_swap] = srcs
    return assign


@torch.no_grad()
def iia(
    site: InterventionSite,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    causal_model,
    task: Task,
    *,
    n_batches: int = 8,
    batch_size: int = 64,
    n_sources: int = 2,
    generator: torch.Generator,
    swap_sizes: tuple[int, ...] = (1, 2),
) -> dict[int, float]:
    """Interchange intervention accuracy per swap size.

    For each batch, sample base/sources from ``task``, build an assignment that
    swaps exactly ``swap_size`` variables from distinct sources, run the frozen
    network under the interchange (hard masks), and compare ``argmax`` of N's
    intervened logits with ``argmax`` of H's counterfactual prediction.

    Returns
    -------
    dict[int, float]
        Mapping ``swap_size -> accuracy in [0, 1]``.
    """
    results: dict[int, float] = {}
    device = rotation.matrix.device
    for swap_size in swap_sizes:
        correct = 0
        total = 0
        for _ in range(n_batches):
            batch = task.sample_batch(batch_size, n_sources, layout.k_max, generator)
            batch = _batch_to_device(batch, device)
            assign = _build_assignment(
                batch.base_inputs.shape[0],
                layout.k_max,
                n_sources,
                swap_size,
                generator,
                device,
            )
            batch = _replace_assignment(batch, assign)
            n_logits = interchange(site, rotation, layout, batch, hard=True)
            h_logits = causal_model.counterfactual_predict(
                batch.base_inputs, batch.source_inputs, batch.source_assignment
            )
            correct += int((n_logits.argmax(-1) == h_logits.argmax(-1)).sum())
            total += n_logits.shape[0]
        results[swap_size] = correct / max(total, 1)
    return results


@dataclass
class RecoveryResult:
    """Result of :func:`recovery`.

    Attributes
    ----------
    matrix:
        ``(k_max, k_gt)`` best per-pair agreement accuracy in ``[0, 1]``.
    best_assignment:
        For each ground-truth variable ``j`` (``0..k_gt-1``), the learned
        variable index assigned to it (distinct), or ``-1`` if unassigned.
    best_score:
        Mean agreement over matched ground-truth variables.
    """

    matrix: list[list[float]]
    best_assignment: list[int]
    best_score: float


@torch.no_grad()
def recovery(
    causal_model,
    task: Task,
    *,
    n_samples: int = 512,
    n_sources: int = 2,
    generator: torch.Generator,
) -> RecoveryResult:
    """Recover ground-truth variables from learned ones (up to relabeling).

    Samples inputs, computes learned variable argmax values and
    ``task.gt_variables``.  For every (learned ``i``, gt ``j``) pair, finds the
    best value-relabeling agreement (brute force over ``v!`` relabelings for
    small ``v``), giving a ``(k_max, k_gt)`` matrix.  Then brute-forces the best
    injective assignment of distinct learned variables to gt variables (``k_gt``
    small).
    """
    k_gt = task.k_gt
    if k_gt <= 0:
        raise EvalError("task has no ground-truth variables (k_gt=0)")
    device = _model_device(causal_model)

    batch = task.sample_batch(n_samples, n_sources, causal_model.k_max, generator)
    batch = _batch_to_device(batch, device)
    inputs = batch.base_inputs

    learned = causal_model.variables(inputs).argmax(-1)  # (N, k_max)
    gt = task.gt_variables(inputs).to(device).to(torch.long)  # (N, k_gt)
    v = causal_model.v
    k_max = causal_model.k_max

    matrix = torch.zeros(k_max, k_gt)
    for i in range(k_max):
        li = learned[:, i]
        for j in range(k_gt):
            gj = gt[:, j]
            matrix[i, j] = _best_relabel_agreement(li, gj, v)

    # Brute-force injective assignment of learned vars -> gt vars.
    best_score = -1.0
    best_assign = [-1] * k_gt
    if k_max >= k_gt:
        for combo in permutations(range(k_max), k_gt):
            score = sum(matrix[combo[j], j].item() for j in range(k_gt)) / k_gt
            if score > best_score:
                best_score = score
                best_assign = list(combo)
    else:  # fewer learned than gt: assign what we can
        for combo in permutations(range(k_gt), k_max):
            score = sum(matrix[i, combo[i]].item() for i in range(k_max)) / k_gt
            if score > best_score:
                best_score = score
                best_assign = [-1] * k_gt
                for i in range(k_max):
                    best_assign[combo[i]] = i

    return RecoveryResult(
        matrix=matrix.tolist(),
        best_assignment=best_assign,
        best_score=best_score,
    )


def _best_relabel_agreement(learned: torch.Tensor, gt: torch.Tensor, v: int) -> float:
    """Best agreement of ``learned`` with ``gt`` over value relabelings.

    For small ``v`` (``<= 4``) brute-force all ``v!`` permutations of learned
    values.  Otherwise use a majority-vote mapping (each learned value maps to
    its most common gt value).
    """
    n = learned.shape[0]
    if n == 0:
        return 0.0
    if v <= 4:
        best = 0.0
        for perm in permutations(range(v)):
            perm_t = torch.tensor(perm, device=learned.device)
            mapped = perm_t[learned]
            acc = (mapped == gt).float().mean().item()
            best = max(best, acc)
        return best
    # Majority-vote mapping for larger v.
    mapped = torch.empty_like(gt)
    for val in range(v):
        sel = learned == val
        if sel.any():
            majority = torch.mode(gt[sel]).values
            mapped[sel] = majority
    return (mapped == gt).float().mean().item()


@torch.no_grad()
def effective_k(
    site: InterventionSite,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    task: Task,
    *,
    n_batches: int = 4,
    batch_size: int = 64,
    n_sources: int = 1,
    generator: torch.Generator,
    flip_threshold: float = 0.02,
) -> int:
    """Number of "live" variables.

    A variable ``i`` counts if its hard mask has width ``>= 1`` dim **and** a
    single-variable swap of just ``i`` flips N's output on more than
    ``flip_threshold`` of pairs (compared to the un-intervened output).
    """
    device = rotation.matrix.device
    widths = layout.hard_widths().tolist()
    live = 0
    for i in range(layout.k_max):
        if widths[i] < 1:
            continue
        flipped = 0
        total = 0
        for _ in range(n_batches):
            batch = task.sample_batch(batch_size, max(n_sources, 1), layout.k_max, generator)
            batch = _batch_to_device(batch, device)
            b = batch.base_inputs.shape[0]
            assign = torch.full((b, layout.k_max), -1, dtype=torch.long, device=device)
            assign[:, i] = 0  # swap only variable i from source 0
            batch_i = _replace_assignment(batch, assign)
            base_logits = site.logits(batch.base_inputs)
            swapped_logits = interchange(site, rotation, layout, batch_i, hard=True)
            flipped += int((base_logits.argmax(-1) != swapped_logits.argmax(-1)).sum())
            total += b
        if total and flipped / total > flip_threshold:
            live += 1
    return live


# -- small helpers --------------------------------------------------------


def _model_device(causal_model) -> torch.device:
    for p in causal_model.parameters():
        return p.device
    return torch.device("cpu")


def _batch_to_device(batch: InterventionBatch, device: torch.device) -> InterventionBatch:
    return InterventionBatch(
        base_inputs=batch.base_inputs.to(device),
        source_inputs=batch.source_inputs.to(device),
        source_assignment=batch.source_assignment.to(device),
        base_labels=batch.base_labels.to(device),
        source_labels=batch.source_labels.to(device),
    )


def _replace_assignment(
    batch: InterventionBatch, assign: torch.Tensor
) -> InterventionBatch:
    return InterventionBatch(
        base_inputs=batch.base_inputs,
        source_inputs=batch.source_inputs,
        source_assignment=assign,
        base_labels=batch.base_labels,
        source_labels=batch.source_labels,
    )
