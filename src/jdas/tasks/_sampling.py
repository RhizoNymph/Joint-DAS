"""Shared interchange-intervention sampling utilities.

The ``source_assignment`` convention (see :mod:`jdas.types`): shape ``(B, k_max)``
long with entries in ``[-1, m)``. Entry ``i == -1`` means "do not swap variable
``i``"; entry ``i == j >= 0`` means "take variable ``i`` from source ``j``".
"""

from __future__ import annotations

import torch


def sample_source_assignment(
    batch_size: int,
    n_sources: int,
    k_max: int,
    k_gt: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample a ``(B, k_max)`` interchange assignment tensor.

    For each example:

    - Sample ``|I|`` in ``{1, 2}`` with probability ``{0.5, 0.5}`` (capped at
      the number of usable variable slots ``k = min(k_max, k_gt)`` and at
      ``n_sources`` so distinct sources are available for ``|I| == 2``).
    - Choose ``|I|`` distinct variable slots uniformly among the first ``k``
      slots.
    - Assign distinct source indices in ``[0, n_sources)`` to the swapped slots
      (distinct required when ``|I| == 2``). All other entries are ``-1``.

    Args:
        batch_size: number of examples ``B``.
        n_sources: number of available sources ``m`` (``>= 1``).
        k_max: number of variable slots in the alignment.
        k_gt: number of ground-truth variables; only the first
            ``min(k_max, k_gt)`` slots are ever swapped so interventions target
            meaningful variables.
        generator: RNG.
        device: output device.

    Returns:
        ``(B, k_max)`` long tensor in ``[-1, n_sources)``.
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be >= 1, got {n_sources}")
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")

    k = min(k_max, k_gt)
    # cap |I| at the number of usable slots and (for distinct sources) at m
    max_i = min(2, k, max(1, n_sources))

    assignment = torch.full(
        (batch_size, k_max), -1, dtype=torch.long, device=device
    )

    # decide |I| per example
    if max_i >= 2:
        size_i = torch.where(
            torch.rand(batch_size, generator=generator, device=device) < 0.5,
            torch.tensor(2, device=device),
            torch.tensor(1, device=device),
        )
    else:
        size_i = torch.ones(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        n_swap = int(size_i[b].item())
        # choose n_swap distinct variable slots among the first k
        var_perm = torch.randperm(k, generator=generator, device=device)[:n_swap]
        # choose distinct source indices for the swapped slots
        src_perm = torch.randperm(n_sources, generator=generator, device=device)[
            :n_swap
        ]
        assignment[b, var_perm] = src_perm
    return assignment
