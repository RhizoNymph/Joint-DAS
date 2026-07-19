"""Per-variable hard-concrete (L0) gates for joint-DAS minimality.

An intrinsic Occam mechanism: one stochastic gate per causal-variable slot with
an L0-style cost, so the number of *live* variables is learned rather than fixed
at ``k_max``.  Implements the hard-concrete distribution of Louizos et al.
(2017, "Learning Sparse Neural Networks through L0 Regularization").

Gate values ``g_i in [0, 1]`` are consumed by BOTH sides of the alignment (see
:mod:`jdas.training` for the coupling that enforces the same sample on both):

- **N-side** (:class:`jdas.rotation.SubspaceLayout`): effective width
  ``w_eff_i = g_i * w_i`` — a closed gate removes variable ``i``'s subspace from
  every interchange swap.
- **H-side** (:class:`jdas.causal_model.LearnedCausalModel`): the discretized
  value one-hot is masked by ``hard(g_i)`` with a straight-through gradient, so a
  gated-off variable is constant-0 in H.

Design note (why hard-concrete and not the width clamp)
-------------------------------------------------------
The Night-2 width clamp caused gradient death once a variable's width
saturated.  The hard-concrete penalty keeps a nonzero gradient on ``log_alpha``
whenever a gate is not fully saturated, and the stretch ``(gamma, zeta)`` lets
the sampled gate reach *exactly* 0/1 while ``log_alpha`` still receives gradient
through the sample distribution.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GateError(ValueError):
    """Raised for invalid gate configuration."""


class VariableGates(nn.Module):
    """Hard-concrete gates, one per causal variable.

    Parameters
    ----------
    k_max:
        Number of variable slots (gates).
    init:
        Initial value of every ``log_alpha`` (default ``+2.0`` ~= 0.88 open, so
        training starts with all variables available).  A value below ``-0.4``
        (roughly) starts gates *closed* (``g_det <= 0.5``); the gate learning
        rate (see :class:`jdas.training.JointConfig`) controls how fast this
        moves during training.
    init_log_alpha:
        Deprecated alias for ``init`` (kept so existing callers/tests keep
        working).  If given, it overrides ``init``.
    beta, gamma, zeta:
        Hard-concrete constants (defaults per the spec: ``2/3``, ``-0.1``,
        ``1.1``).  ``gamma < 0 < zeta`` and ``zeta > 1`` give the stretch that
        lets gates reach exactly 0 and 1.
    """

    def __init__(
        self,
        k_max: int,
        *,
        init: float = 2.0,
        init_log_alpha: float | None = None,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ) -> None:
        super().__init__()
        if k_max <= 0:
            raise GateError(f"k_max must be positive, got {k_max}")
        if beta <= 0.0:
            raise GateError(f"beta must be positive, got {beta}")
        if not (gamma < 0.0 < zeta):
            raise GateError(f"require gamma < 0 < zeta, got gamma={gamma}, zeta={zeta}")
        self.k_max = k_max
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        init_value = init if init_log_alpha is None else init_log_alpha
        self.log_alpha = nn.Parameter(torch.full((k_max,), float(init_value)))

    # -- sampling ---------------------------------------------------------

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Train-time stochastic gate ``g in [0, 1]^{k_max}``.

        ``u ~ U(0,1)``,
        ``s = sigmoid((log u - log(1-u) + log_alpha) / beta)``,
        ``g = clamp(s * (zeta - gamma) + gamma, 0, 1)``.

        The clamp is what lets ``g`` reach exactly 0/1; the un-clamped
        ``stretched`` still carries gradient into ``log_alpha`` for
        non-saturated entries.
        """
        la = self.log_alpha
        u = torch.rand(la.shape, generator=generator, device=la.device, dtype=la.dtype)
        # Keep u strictly interior so log/logit are finite.
        eps = torch.finfo(la.dtype).tiny
        u = u.clamp(min=eps, max=1.0 - torch.finfo(la.dtype).eps)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / self.beta)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def deterministic(self) -> torch.Tensor:
        """Eval-time deterministic gate ``g_det in [0, 1]^{k_max}``.

        ``g_det = clamp(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)``.
        """
        s = torch.sigmoid(self.log_alpha)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    # -- liveness ---------------------------------------------------------

    def live_mask(self) -> torch.Tensor:
        """Boolean ``(k_max,)`` mask of live variables (``g_det > 0.5``)."""
        return self.deterministic() > 0.5

    def gated_k(self) -> int:
        """Parameter-based live count (number of ``g_det > 0.5``)."""
        return int(self.live_mask().sum().item())

    def live_indices(self) -> list[int]:
        """Indices of live variables as a Python list."""
        return [i for i, live in enumerate(self.live_mask().tolist()) if live]

    # -- penalty ----------------------------------------------------------

    def penalty(self) -> torch.Tensor:
        """L0 penalty (expected number of open gates), a differentiable scalar.

        ``L_gate = sum_i sigmoid(log_alpha_i - beta * log(-gamma / zeta))``.
        """
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).sum()

    # -- straight-through hard gate (H-side value mask) -------------------

    @staticmethod
    def hard(g: torch.Tensor) -> torch.Tensor:
        """Straight-through hard gate: ``(g > 0.5)`` forward, gradient of ``g``.

        Used to mask H's discretized variable values: ``v_used_i = hard(g_i) *
        v_i``.  Forward is 0/1; ``d hard / d g == 1`` so ``log_alpha`` still
        receives gradient from the H-side value mask.
        """
        h = (g > 0.5).to(g.dtype)
        return h - g.detach() + g
