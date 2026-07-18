"""Orthogonal rotation and learned subspace layout for DAS-style alignment.

This module provides the two geometric ingredients of Distributed Alignment
Search:

- :class:`OrthogonalRotation` -- a learnable orthogonal matrix ``Q`` of shape
  ``(d, d)`` that rotates the site hidden vector into an "aligned" basis.
- :class:`SubspaceLayout` -- a learnable partition of the ``d`` rotated
  coordinates into ``k_max`` contiguous, disjoint blocks (one per high-level
  variable), with a continuous (annealed) relaxation for training and a hard
  boolean version for evaluation.

Orientation convention
----------------------
We store ``Q`` as the weight matrix of an ``nn.Linear(d, d, bias=False)``
parametrized to be orthogonal.  For a hidden row-vector batch ``h`` of shape
``(B, d)`` we define::

    rotate(h)   = h @ Q.T      (== Linear(h), the "aligned" coordinates)
    unrotate(r) = r @ Q        (inverse, since Q is orthogonal: Q.T @ Q = I)

so that ``unrotate(rotate(h)) == h``.  Aligned coordinate ``p`` is
``(h @ Q.T)[:, p] = h @ Q[p]`` (the ``p``-th *row* of ``Q``).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal


class RotationError(ValueError):
    """Raised for invalid rotation / subspace-layout configuration."""


class OrthogonalRotation(nn.Module):
    """A learnable orthogonal matrix ``Q`` of shape ``(d, d)``.

    The matrix is the weight of an orthogonally-parametrized linear layer, so
    ``Q @ Q.T == I`` holds (up to numerical error) for *every* value of the
    underlying raw parameters throughout optimization.

    Parameters
    ----------
    d:
        Dimensionality of the hidden vector at the intervention site.
    freeze:
        If ``True`` the rotation is frozen at its (random) initialization; used
        for the "random rotation" control from the design doc.
    """

    def __init__(self, d: int, *, freeze: bool = False) -> None:
        super().__init__()
        if d <= 0:
            raise RotationError(f"rotation dim d must be positive, got {d}")
        self.d = d
        linear = nn.Linear(d, d, bias=False)
        # `orthogonal` reparametrizes linear.weight so it is always orthogonal.
        self._linear = orthogonal(linear)
        if freeze:
            for p in self._linear.parameters():
                p.requires_grad_(False)
        self.frozen = freeze

    @property
    def matrix(self) -> torch.Tensor:
        """The current orthogonal matrix ``Q`` of shape ``(d, d)``."""
        return self._linear.weight

    def set_matrix(self, q: torch.Tensor) -> None:
        """Set the rotation to a specific orthogonal matrix ``q`` of shape
        ``(d, d)``.

        Uses the orthogonal parametrization's ``right_inverse`` (via attribute
        assignment) so the underlying raw parameter is set consistently.  ``q``
        must be (approximately) orthogonal.
        """
        if q.shape != (self.d, self.d):
            raise RotationError(f"set_matrix expects ({self.d}, {self.d}), got {tuple(q.shape)}")
        self._linear.weight = q.to(self._linear.weight.dtype)

    def rotate(self, h: torch.Tensor) -> torch.Tensor:
        """Rotate hidden vectors into the aligned basis.

        Parameters
        ----------
        h:
            Tensor of shape ``(..., d)``.

        Returns
        -------
        torch.Tensor
            ``h @ Q.T`` of shape ``(..., d)``.
        """
        if h.shape[-1] != self.d:
            raise RotationError(
                f"expected last dim {self.d}, got hidden of shape {tuple(h.shape)}"
            )
        return self._linear(h)

    def unrotate(self, r: torch.Tensor) -> torch.Tensor:
        """Map aligned coordinates back to the original hidden basis.

        Parameters
        ----------
        r:
            Tensor of shape ``(..., d)`` in the aligned basis.

        Returns
        -------
        torch.Tensor
            ``r @ Q`` of shape ``(..., d)``.
        """
        if r.shape[-1] != self.d:
            raise RotationError(
                f"expected last dim {self.d}, got rotated of shape {tuple(r.shape)}"
            )
        return r @ self.matrix


class SubspaceLayout(nn.Module):
    """Learnable contiguous disjoint blocks over the ``d`` rotated coordinates.

    Each high-level variable ``i`` (``0 <= i < k_max``) owns the contiguous set
    of aligned coordinates ``[c_{i-1}, c_i)`` where the boundaries are the
    cumulative sums of non-negative widths ``w_i`` clamped to ``d``.  Because the
    blocks are cumulative and non-overlapping by construction, masks of different
    variables never overlap.

    Width parameterization
    ----------------------
    Two modes for ``w_i``:

    - **Unbounded** (``max_width is None``, the default): ``w_i =
      softplus(raw_i)``, initialized so ``w_i == init_width``.  This is the
      historical behavior.
    - **Bounded** (``max_width`` set): ``w_i = max_width * sigmoid(raw_i)``,
      initialized so ``w_i ~= init_width`` (requires ``init_width <
      max_width``).  A hard per-variable width cap is then guaranteed:
      ``w_i < max_width`` for every value of ``raw_i``, so no single variable can
      absorb an arbitrarily large block regardless of optimization pressure.  The
      cap addresses the LM-scale collapse where one variable owns the whole
      residual stream.

    Two mask flavours are produced:

    - :meth:`soft_masks` -- a smooth ``(k_max, d)`` relaxation using a product
      of sigmoids around each block, controlled by ``temperature`` (annealed
      externally).  Used for differentiable training.
    - :meth:`hard_masks` -- a boolean ``(k_max, d)`` partition using the integer
      boundaries; used for evaluation.

    Parameters
    ----------
    d:
        Number of rotated coordinates.
    k_max:
        Maximum number of variables (blocks).
    init_width:
        Initial expected width (in dims) of each block.  ``raw`` is initialized
        so that ``widths() == init_width`` (approximately, in bounded mode).
    max_width:
        If ``None`` (default) widths are unbounded (``softplus``).  If set, each
        width is capped strictly below ``max_width`` via
        ``max_width * sigmoid(raw)``; requires ``0 < init_width < max_width``.
    min_temp, max_temp:
        Clamp range for the mask temperature.
    """

    def __init__(
        self,
        d: int,
        k_max: int,
        *,
        init_width: float = 1.0,
        max_width: float | None = None,
        min_temp: float = 0.05,
        max_temp: float = 5.0,
    ) -> None:
        super().__init__()
        if d <= 0 or k_max <= 0:
            raise RotationError(f"d and k_max must be positive, got d={d}, k_max={k_max}")
        if not (0.0 < min_temp <= max_temp):
            raise RotationError(
                f"require 0 < min_temp <= max_temp, got {min_temp}, {max_temp}"
            )
        if init_width <= 0.0:
            raise RotationError(f"init_width must be positive, got {init_width}")
        if max_width is not None and not (init_width < max_width):
            raise RotationError(
                f"require init_width < max_width, got init_width={init_width}, "
                f"max_width={max_width}"
            )
        self.d = d
        self.k_max = k_max
        self.max_width = None if max_width is None else float(max_width)
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)

        if self.max_width is None:
            # Inverse-softplus of init_width so softplus(raw_widths) == init_width.
            inv = torch.log(torch.expm1(torch.tensor(float(init_width))))
        else:
            # Inverse-sigmoid (logit) of init_width / max_width so that
            # max_width * sigmoid(raw_widths) == init_width.
            frac = float(init_width) / self.max_width
            inv = torch.logit(torch.tensor(frac))
        self.raw_widths = nn.Parameter(torch.full((k_max,), float(inv)))
        # Mutable temperature (buffer so it moves with .to(device) and is saved).
        self.register_buffer("_temperature", torch.tensor(1.0))

    # -- temperature ------------------------------------------------------

    @property
    def temperature(self) -> float:
        return float(self._temperature)

    def set_temperature(self, tau: float) -> None:
        """Set the mask relaxation temperature, clamped to ``[min_temp, max_temp]``."""
        tau = max(self.min_temp, min(self.max_temp, float(tau)))
        self._temperature.fill_(tau)

    # -- boundaries -------------------------------------------------------

    def widths(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Non-negative block widths ``(k_max,)``.

        Unbounded mode: ``softplus(raw)``.  Bounded mode: ``max_width *
        sigmoid(raw)`` (strictly ``< max_width``).

        If ``gate`` (shape ``(k_max,)`` in ``[0, 1]``) is given, the returned
        *effective* widths are gate-scaled: ``w_eff_i = g_i * w_i``.  A closed
        gate (``g_i == 0``) yields a zero-width block, removing that variable's
        subspace from every interchange swap.
        """
        if self.max_width is None:
            w = torch.nn.functional.softplus(self.raw_widths)
        else:
            w = self.max_width * torch.sigmoid(self.raw_widths)
        if gate is not None:
            w = w * self._check_gate(gate)
        return w

    def _check_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape != (self.k_max,):
            raise RotationError(
                f"gate shape {tuple(gate.shape)} != ({self.k_max},)"
            )
        return gate.to(self.raw_widths.dtype)

    def boundaries(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Cumulative boundaries ``c_0..c_{k_max}`` of shape ``(k_max + 1,)``.

        ``c_0 == 0`` and each ``c_i`` is clamped to ``[0, d]``.  ``c_i`` is the
        (soft, real-valued) right edge of block ``i-1`` / left edge of block
        ``i``.  With ``gate`` supplied, gate-scaled effective widths are used.
        """
        w = self.widths(gate=gate)
        cum = torch.cumsum(w, dim=0).clamp(max=float(self.d))
        zero = cum.new_zeros(1)
        return torch.cat([zero, cum], dim=0)

    # -- masks ------------------------------------------------------------

    def soft_masks(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable ``(k_max, d)`` masks in ``[0, 1]``.

        Block ``i`` spans ``[c_i, c_{i+1})``.  For coordinate position ``pos``
        (``0..d-1`` at cell centres ``pos + 0.5``) the mask is::

            sigmoid((pos+0.5 - c_i)/tau) * sigmoid((c_{i+1} - (pos+0.5))/tau)

        which is ~1 strictly inside the block and ~0 outside, with a soft ramp
        of width ~tau at the boundaries.  Because blocks are disjoint and
        adjacent share the boundary ``c_i``, the columns sum to <= 1.

        With ``gate`` supplied, gate-scaled effective widths set the boundaries
        (a closed gate gives a zero-width block) *and* the per-row mass is scaled
        by ``g_i`` so a closed gate's mask is uniformly ~0 (not merely narrow),
        which keeps the interchange contribution a true no-op.
        """
        c = self.boundaries(gate=gate)  # (k_max + 1,)
        tau = self._temperature.clamp(min=self.min_temp, max=self.max_temp)
        pos = torch.arange(self.d, device=c.device, dtype=c.dtype) + 0.5  # (d,)
        left = c[:-1].unsqueeze(1)  # (k_max, 1) start of each block
        right = c[1:].unsqueeze(1)  # (k_max, 1) end of each block
        p = pos.unsqueeze(0)  # (1, d)
        m = torch.sigmoid((p - left) / tau) * torch.sigmoid((right - p) / tau)
        if gate is not None:
            # Scale each row's mass by g_i so a closed gate is a uniform ~0 mask
            # (a zero-width block already collapses the boundaries, but scaling
            # makes the no-op exact and keeps the gate in the autograd path).
            m = m * self._check_gate(gate).unsqueeze(1)
        return m

    def hard_masks(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Boolean ``(k_max, d)`` disjoint partition using integer boundaries.

        Coordinate ``pos`` belongs to block ``i`` iff ``c_i <= pos < c_{i+1}``
        after rounding the boundaries to integers.  Guaranteed disjoint.  With
        ``gate`` supplied, a variable whose gate is closed (``g_i <= 0.5``) gets
        an empty (all-``False``) row.
        """
        c = self.boundaries(gate=gate).detach()
        # Round boundaries to ints, keep monotonic non-decreasing and in [0, d].
        edges = torch.clamp(torch.round(c), min=0.0, max=float(self.d)).long()
        edges = torch.cummax(edges, dim=0).values  # enforce monotonicity
        pos = torch.arange(self.d, device=c.device).long()  # (d,)
        left = edges[:-1].unsqueeze(1)  # (k_max, 1)
        right = edges[1:].unsqueeze(1)  # (k_max, 1)
        p = pos.unsqueeze(0)  # (1, d)
        masks = (p >= left) & (p < right)
        if gate is not None:
            # Zero out rows whose gate is closed (g_i <= 0.5); belt-and-braces
            # with the boundary collapse above.
            live = (self._check_gate(gate) > 0.5).unsqueeze(1)
            masks = masks & live
        return masks

    def hard_widths(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Integer width (in dims) of each block, shape ``(k_max,)`` long."""
        return self.hard_masks(gate=gate).sum(dim=1)

    def total_aligned_dims(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable scalar: total aligned dims ``= c_{k_max}``.

        Equal to the sum of (gate-scaled) soft widths clamped to ``d``; used for
        the sparsity penalty ``L_sparse = lambda * total_aligned_dims / d``.
        """
        return self.boundaries(gate=gate)[-1]
