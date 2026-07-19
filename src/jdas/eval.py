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


def _build_assignment_over(
    b: int,
    k_max: int,
    live: list[int],
    n_sources: int,
    swap_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """``(b, k_max)`` assignment swapping ``swap_size`` variables drawn only from
    the ``live`` index set (each from a distinct source).

    Variables outside ``live`` are never swapped (their column stays ``-1``), so
    dead-variable no-op swaps cannot inflate the metric.
    """
    if swap_size > len(live):
        raise EvalError(
            f"swap_size {swap_size} > number of live variables {len(live)}"
        )
    if swap_size > n_sources:
        raise EvalError(
            f"swap_size {swap_size} needs >= {swap_size} sources, got {n_sources}"
        )
    live_t = torch.tensor(live, dtype=torch.long, device=device)
    assign = torch.full((b, k_max), -1, dtype=torch.long, device=device)
    n_live = live_t.shape[0]
    for row in range(b):
        pick = torch.randperm(n_live, generator=generator, device=device)[:swap_size]
        vars_to_swap = live_t[pick]
        src_perm = torch.randperm(n_sources, generator=generator, device=device)
        assign[row, vars_to_swap] = src_perm[:swap_size]
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
    # Swap sizes beyond the model's variable count are undefined; skip them
    # (e.g. a k=1 output-copy baseline has no |I|=2 interventions).
    swap_sizes = tuple(s for s in swap_sizes if s <= layout.k_max)
    if not swap_sizes:
        raise EvalError(f"no valid swap sizes for k_max {layout.k_max}")
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


@torch.no_grad()
def iia_live(
    site: InterventionSite,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    causal_model,
    task: Task,
    gates,
    *,
    n_batches: int = 8,
    batch_size: int = 64,
    n_sources: int = 2,
    generator: torch.Generator,
    swap_sizes: tuple[int, ...] = (1, 2),
) -> dict[int, float | None]:
    """Live-restricted interchange intervention accuracy.

    Identical to :func:`iia` but (a) swaps are sampled only over *live*
    variables (``gates.live_mask()``) and (b) the deterministic gate
    ``gates.deterministic()`` is passed to BOTH N's interchange and H's
    counterfactual prediction, so the same dead-variable no-op holds on both
    sides.  This is the honest headline metric for gated runs: all-variable IIA
    is inflated by no-op swaps of dead variables.

    Returns
    -------
    dict[int, float | None]
        ``swap_size -> accuracy`` for each requested size that is applicable
        (``swap_size <= number of live variables``); inapplicable sizes map to
        ``None`` (JSON null) rather than a fake ``0.0``.
    """
    from .causal_model import LearnedCausalModel

    device = rotation.matrix.device
    g_det = gates.deterministic().to(device)
    live = gates.live_indices()
    # The N-side alignment always sees the gate.  A fixed H has no gates (all
    # variables live by construction), so it receives gate=None; a learned H is
    # value-masked by the same gate for the symmetric dead-variable no-op.
    h_gate = g_det if isinstance(causal_model, LearnedCausalModel) else None
    results: dict[int, float | None] = {}
    for swap_size in swap_sizes:
        if swap_size > len(live):
            results[swap_size] = None
            continue
        correct = 0
        total = 0
        for _ in range(n_batches):
            batch = task.sample_batch(batch_size, n_sources, layout.k_max, generator)
            batch = _batch_to_device(batch, device)
            assign = _build_assignment_over(
                batch.base_inputs.shape[0],
                layout.k_max,
                live,
                n_sources,
                swap_size,
                generator,
                device,
            )
            batch = _replace_assignment(batch, assign)
            n_logits = interchange(site, rotation, layout, batch, hard=True, gate=g_det)
            h_logits = causal_model.counterfactual_predict(
                batch.base_inputs, batch.source_inputs, batch.source_assignment,
                gate=h_gate,
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
    live_indices: list[int] | None = None,
) -> RecoveryResult:
    """Recover ground-truth variables from learned ones (up to relabeling).

    Samples inputs, computes learned variable argmax values and
    ``task.gt_variables``.  For every (learned ``i``, gt ``j``) pair, finds the
    best value-relabeling agreement (brute force over ``v!`` relabelings for
    small ``v``), giving a ``(k_max, k_gt)`` matrix.  Then brute-forces the best
    injective assignment of distinct learned variables to gt variables (``k_gt``
    small).

    ``live_indices`` (gated runs): if given, only those learned variables are
    candidates for assignment to ground-truth variables (dead variables are
    ignored), so recovery is computed over live variables only.  The full
    ``(k_max, k_gt)`` agreement matrix is still returned for inspection.
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

    # Candidate learned-variable indices (all, or the live subset for gated runs).
    candidates = list(range(k_max)) if live_indices is None else list(live_indices)

    # Brute-force injective assignment of candidate learned vars -> gt vars.
    best_score = -1.0
    best_assign = [-1] * k_gt
    if len(candidates) >= k_gt:
        for combo in permutations(candidates, k_gt):
            score = sum(matrix[combo[j], j].item() for j in range(k_gt)) / k_gt
            if score > best_score:
                best_score = score
                best_assign = list(combo)
    else:  # fewer candidate learned vars than gt: assign what we can
        n_cand = len(candidates)
        for combo in permutations(range(k_gt), n_cand):
            score = sum(matrix[candidates[i], combo[i]].item() for i in range(n_cand)) / k_gt
            if score > best_score:
                best_score = score
                best_assign = [-1] * k_gt
                for i in range(n_cand):
                    best_assign[combo[i]] = candidates[i]

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
