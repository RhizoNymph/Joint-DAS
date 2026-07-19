"""Interchange intervention machinery.

Given a frozen :class:`~jdas.types.InterventionSite`, an
:class:`~jdas.rotation.OrthogonalRotation`, and a
:class:`~jdas.rotation.SubspaceLayout`, :func:`interchange` runs the network on
a base input while swapping, for each variable ``i`` selected by
``source_assignment``, the aligned-subspace content of the base hidden with that
of a chosen source hidden.

Gradient flow
-------------
The source hiddens depend on ``Q`` (through :meth:`rotate`) and must *not* be
detached, so gradients reach the rotation from both base and source paths.  The
site's own weights are frozen by the site implementation; here we only require
that the site keeps the autograd graph from ``hidden`` through
``logits_with_hidden`` back to the rotation/layout parameters.
"""

from __future__ import annotations

import torch

from .causal_model import _check_assignment
from .rotation import OrthogonalRotation, SubspaceLayout
from .types import InterventionBatch, InterventionSite


class InterventionError(ValueError):
    """Raised for shape/param errors in the interchange routine."""


def interchange(
    site: InterventionSite,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    batch: InterventionBatch,
    *,
    hard: bool = False,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one batch of interchange interventions and return N's logits.

    Steps (shapes: ``B`` batch, ``m`` sources, ``d`` site dim, ``k`` k_max):

    1. Rotate base hidden ``r_b = rotate(h(base))`` -> ``(B, d)``.
    2. Rotate all source hiddens ``r_s`` -> ``(B, m, d)`` (one batched forward
       over ``B*m`` inputs).
    3. Build a per-example, per-variable swap using the layout masks and
       ``source_assignment``:  for each variable ``i`` with assignment ``j_i >=
       0`` overwrite its masked aligned coordinates with those of source
       ``j_i``.
    4. Unrotate the mixed aligned vector and rerun the site from the site.

    Parameters
    ----------
    site:
        Frozen network with a hidden intervention site.
    rotation, layout:
        Trainable rotation and subspace layout.
    batch:
        The interchange batch (base/source inputs + assignment).
    hard:
        If ``True`` use boolean :meth:`SubspaceLayout.hard_masks` (evaluation);
        else the differentiable :meth:`soft_masks`.
    gate:
        Optional ``(k_max,)`` per-variable gate in ``[0, 1]``.  Effective widths
        ``w_eff_i = g_i * w_i`` feed mask construction, so a closed gate removes
        that variable's subspace from the swap (a no-op on N).  The *same* gate
        sample must be passed to H's counterfactual prediction within one
        forward/loss so the dead-variable no-op is symmetric (see
        :mod:`jdas.training`).

    Returns
    -------
    torch.Tensor
        ``(B, n_labels)`` logits of the frozen network under the intervention.
    """
    base_inputs = batch.base_inputs
    source_inputs = batch.source_inputs
    assign = batch.source_assignment.to(torch.long)

    b = base_inputs.shape[0]
    m = source_inputs.shape[1]
    k = layout.k_max
    d = layout.d
    _check_assignment(assign, b, k, m)

    # 1. base rotated hidden (B, d)
    r_b = rotation.rotate(site.hidden(base_inputs))
    if r_b.shape != (b, d):
        raise InterventionError(f"base rotated hidden {tuple(r_b.shape)} != ({b}, {d})")

    # 2. source rotated hiddens (B, m, d) via a single (B*m, ...) forward.
    src_flat = source_inputs.reshape(b * m, *source_inputs.shape[2:])
    r_s = rotation.rotate(site.hidden(src_flat)).reshape(b, m, d)

    # 3. masks (k, d)
    masks = (
        layout.hard_masks(gate=gate).to(r_b.dtype)
        if hard
        else layout.soft_masks(gate=gate)
    )

    # Per (b, i): the mask if swapped, else zeros.  swap indicator (B, k).
    swap = (assign >= 0)  # (B, k) bool
    gather_j = assign.clamp(min=0)  # (B, k), -1 -> 0 placeholder

    # Gather each variable's chosen source rotated hidden: (B, k, d).
    idx = gather_j.unsqueeze(-1).expand(b, k, d)  # (B, k, d)
    chosen_src = torch.gather(
        r_s, 1, idx
    )  # gather along m -> (B, k, d): src for each var i

    # Broadcast masks to (B, k, d) and gate by swap indicator.
    masks_bkd = masks.unsqueeze(0).expand(b, k, d)  # (B, k, d)
    gate = masks_bkd * swap.unsqueeze(-1).to(masks.dtype)  # (B, k, d)

    # Total mask taken from *some* source, and the summed source contribution.
    total_swap_mask = gate.sum(dim=1)  # (B, d) -- in [0, 1] since blocks disjoint
    src_contrib = (gate * chosen_src).sum(dim=1)  # (B, d)

    r_new = r_b * (1.0 - total_swap_mask) + src_contrib  # (B, d)

    # 4. unrotate and rerun the frozen site.
    h_new = rotation.unrotate(r_new)
    return site.logits_with_hidden(base_inputs, h_new)
