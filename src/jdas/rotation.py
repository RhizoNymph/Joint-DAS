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
    cumulative sums of non-negative widths ``w_i = softplus(raw_i)`` clamped to
    ``d``.  Because the blocks are cumulative and non-overlapping by
    construction, masks of different variables never overlap.

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
        so that ``softplus(raw) == init_width``.
    min_temp, max_temp:
        Clamp range for the mask temperature.
    """

    def __init__(
        self,
        d: int,
        k_max: int,
        *,
        init_width: float = 1.0,
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
        self.d = d
        self.k_max = k_max
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)

        # Inverse-softplus of init_width so softplus(raw_widths) == init_width.
        inv = torch.log(torch.expm1(torch.tensor(float(init_width))))
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

    def widths(self) -> torch.Tensor:
        """Non-negative block widths ``(k_max,)`` via softplus of raw params."""
        return torch.nn.functional.softplus(self.raw_widths)

    def boundaries(self) -> torch.Tensor:
        """Cumulative boundaries ``c_0..c_{k_max}`` of shape ``(k_max + 1,)``.

        ``c_0 == 0`` and each ``c_i`` is clamped to ``[0, d]``.  ``c_i`` is the
        (soft, real-valued) right edge of block ``i-1`` / left edge of block
        ``i``.
        """
        w = self.widths()
        cum = torch.cumsum(w, dim=0).clamp(max=float(self.d))
        zero = cum.new_zeros(1)
        return torch.cat([zero, cum], dim=0)

    # -- masks ------------------------------------------------------------

    def soft_masks(self) -> torch.Tensor:
        """Differentiable ``(k_max, d)`` masks in ``[0, 1]``.

        Block ``i`` spans ``[c_i, c_{i+1})``.  For coordinate position ``pos``
        (``0..d-1`` at cell centres ``pos + 0.5``) the mask is::

            sigmoid((pos+0.5 - c_i)/tau) * sigmoid((c_{i+1} - (pos+0.5))/tau)

        which is ~1 strictly inside the block and ~0 outside, with a soft ramp
        of width ~tau at the boundaries.  Because blocks are disjoint and
        adjacent share the boundary ``c_i``, the columns sum to <= 1.
        """
        c = self.boundaries()  # (k_max + 1,)
        tau = self._temperature.clamp(min=self.min_temp, max=self.max_temp)
        pos = torch.arange(self.d, device=c.device, dtype=c.dtype) + 0.5  # (d,)
        left = c[:-1].unsqueeze(1)  # (k_max, 1) start of each block
        right = c[1:].unsqueeze(1)  # (k_max, 1) end of each block
        p = pos.unsqueeze(0)  # (1, d)
        m = torch.sigmoid((p - left) / tau) * torch.sigmoid((right - p) / tau)
        return m

    def hard_masks(self) -> torch.Tensor:
        """Boolean ``(k_max, d)`` disjoint partition using integer boundaries.

        Coordinate ``pos`` belongs to block ``i`` iff ``c_i <= pos < c_{i+1}``
        after rounding the boundaries to integers.  Guaranteed disjoint.
        """
        c = self.boundaries().detach()
        # Round boundaries to ints, keep monotonic non-decreasing and in [0, d].
        edges = torch.clamp(torch.round(c), min=0.0, max=float(self.d)).long()
        edges = torch.cummax(edges, dim=0).values  # enforce monotonicity
        pos = torch.arange(self.d, device=c.device).long()  # (d,)
        left = edges[:-1].unsqueeze(1)  # (k_max, 1)
        right = edges[1:].unsqueeze(1)  # (k_max, 1)
        p = pos.unsqueeze(0)  # (1, d)
        return (p >= left) & (p < right)

    def hard_widths(self) -> torch.Tensor:
        """Integer width (in dims) of each block, shape ``(k_max,)`` long."""
        return self.hard_masks().sum(dim=1)

    def total_aligned_dims(self) -> torch.Tensor:
        """Differentiable scalar: total aligned dims ``= c_{k_max}``.

        Equal to the sum of soft widths clamped to ``d``; used for the sparsity
        penalty ``L_sparse = lambda * total_aligned_dims / d``.
        """
        return self.boundaries()[-1]
