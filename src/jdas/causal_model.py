"""Learned and fixed high-level causal models H.

A causal model maps a flattened task input to ``k_max`` discrete latent
variables ``Z_1..Z_{k_max}`` (each in ``{0..v-1}``) and decodes them to a label
distribution.  It also supports *counterfactual* prediction: recompute the
label when some variables' values are copied from source inputs.

Two implementations share one interface:

- :class:`LearnedCausalModel` -- trainable per-variable encoder MLPs + decoder,
  with straight-through argmax discretization.
- :class:`FixedCausalModel` -- variables and labels come from ground-truth
  callables; non-trainable.  Used for classic-DAS and wrong-hypothesis
  baselines, and to "freeze" a learned model for refitting.

Discretization (straight-through argmax)
----------------------------------------
Forward pass uses a hard one-hot of ``argmax(logits)``.  The backward pass uses
the gradient of ``softmax(logits / tau_g)`` (temperature ``tau_g`` annealed
externally).  Concretely ``z = hard - soft.detach() + soft`` so that ``z ==
hard`` numerically while ``d z / d logits == d soft / d logits``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class CausalModelError(ValueError):
    """Raised for invalid causal-model configuration or inputs."""


def _straight_through_onehot(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Straight-through one-hot of ``argmax(logits)`` over the last dim.

    Parameters
    ----------
    logits:
        ``(..., v)`` real-valued scores.
    tau:
        Softmax temperature for the backward surrogate (``> 0``).

    Returns
    -------
    torch.Tensor
        ``(..., v)`` one-hot in the forward pass; gradients flow as through
        ``softmax(logits / tau)``.
    """
    soft = torch.softmax(logits / tau, dim=-1)
    idx = torch.argmax(logits, dim=-1)
    hard = torch.nn.functional.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


class LearnedCausalModel(nn.Module):
    """Trainable causal model with per-variable encoders and a small decoder.

    Parameters
    ----------
    input_dim:
        Flattened dimensionality of a task input (``prod(input_shape)``).
    k_max:
        Number of high-level variables.
    v:
        Cardinality of each variable (default binary ``v=2``).
    n_labels:
        Number of task output labels (decoder output size).
    encoder_hidden:
        Hidden width of each per-variable encoder MLP.
    decoder_hidden:
        If ``None`` the decoder is linear ``(k_max*v -> n_labels)``; otherwise a
        one-hidden-layer MLP with this width.
    """

    def __init__(
        self,
        input_dim: int,
        k_max: int,
        v: int = 2,
        n_labels: int = 2,
        *,
        encoder_hidden: int = 64,
        decoder_hidden: int | None = 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or k_max <= 0 or v <= 1 or n_labels <= 0:
            raise CausalModelError(
                f"invalid dims input_dim={input_dim}, k_max={k_max}, v={v}, "
                f"n_labels={n_labels}"
            )
        self.input_dim = input_dim
        self.k_max = k_max
        self.v = v
        self.n_labels = n_labels

        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, encoder_hidden),
                nn.ReLU(),
                nn.Linear(encoder_hidden, v),
            )
            for _ in range(k_max)
        )
        if decoder_hidden is None:
            self.decoder: nn.Module = nn.Linear(k_max * v, n_labels)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(k_max * v, decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, n_labels),
            )
        self.register_buffer("_tau_g", torch.tensor(1.0))

    # -- temperature ------------------------------------------------------

    @property
    def temperature(self) -> float:
        return float(self._tau_g)

    def set_temperature(self, tau: float) -> None:
        """Set the straight-through softmax temperature ``tau_g > 0``."""
        if tau <= 0.0:
            raise CausalModelError(f"temperature must be positive, got {tau}")
        self._tau_g.fill_(float(tau))

    # -- helpers ----------------------------------------------------------

    def _flatten(self, inputs: torch.Tensor) -> torch.Tensor:
        """Flatten inputs to ``(B, input_dim)``."""
        flat = inputs.reshape(inputs.shape[0], -1)
        if flat.shape[1] != self.input_dim:
            raise CausalModelError(
                f"expected flattened input_dim {self.input_dim}, got {flat.shape[1]}"
            )
        return flat

    def variable_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Raw per-variable logits ``(B, k_max, v)``."""
        flat = self._flatten(inputs)
        logits = [enc(flat) for enc in self.encoders]  # each (B, v)
        return torch.stack(logits, dim=1)  # (B, k_max, v)

    def variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot variables ``(B, k_max, v)``."""
        logits = self.variable_logits(inputs)
        return _straight_through_onehot(logits, float(self._tau_g))

    def decode(self, onehots: torch.Tensor) -> torch.Tensor:
        """Decode variable one-hots ``(B, k_max, v)`` to logits ``(B, n_labels)``."""
        b = onehots.shape[0]
        return self.decoder(onehots.reshape(b, self.k_max * self.v))

    def predict(self, inputs: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Clean label logits ``(B, n_labels)`` for ``inputs``.

        With ``gate`` supplied each variable's one-hot is masked by the
        straight-through hard gate (``v_used_i = hard(g_i) * v_i``), so a
        gated-off variable is constant-0 in H.
        """
        return self.decode(_apply_gate(self.variables(inputs), gate))

    def counterfactual_predict(
        self,
        base_inputs: torch.Tensor,
        source_inputs: torch.Tensor,
        source_assignment: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Counterfactual label logits ``(B, n_labels)``.

        Compute base variables ``z(base)`` and replace ``z_i`` with the source's
        value ``z_i(source_j)`` wherever ``source_assignment[b, i] == j >= 0``.

        Parameters
        ----------
        base_inputs:
            ``(B, ...)``.
        source_inputs:
            ``(B, m, ...)`` -- ``m`` sources per base.
        source_assignment:
            ``(B, k_max)`` long in ``[-1, m)``; ``-1`` = keep base.

        Returns
        -------
        torch.Tensor
            ``(B, n_labels)`` decoded from the mixed variable one-hots.
        """
        b = base_inputs.shape[0]
        m = source_inputs.shape[1]
        _check_assignment(source_assignment, b, self.k_max, m)

        base_z = self.variables(base_inputs)  # (B, k_max, v)
        # Source variables: flatten (B, m, ...) -> (B*m, ...) for one pass.
        src_flat = source_inputs.reshape(b * m, *source_inputs.shape[2:])
        src_z = self.variables(src_flat).reshape(b, m, self.k_max, self.v)

        # For each (b, i) gather the chosen source's one-hot (or base if -1).
        assign = source_assignment.to(torch.long)  # (B, k_max)
        gather_j = assign.clamp(min=0)  # (B, k_max); -1 -> 0 placeholder
        # src_z: (B, m, k_max, v) -> select along m per (b, i).
        idx = gather_j.unsqueeze(1).unsqueeze(-1).expand(b, 1, self.k_max, self.v)
        chosen_src = torch.gather(src_z, 1, idx).squeeze(1)  # (B, k_max, v)

        swap = (assign >= 0).unsqueeze(-1).to(base_z.dtype)  # (B, k_max, 1)
        mixed = base_z * (1.0 - swap) + chosen_src * swap
        return self.decode(_apply_gate(mixed, gate))


class FixedCausalModel(nn.Module):
    """Non-trainable causal model backed by ground-truth callables.

    Parameters
    ----------
    gt_variables_fn:
        ``inputs (B, ...) -> (B, k)`` long variable values in ``{0..v-1}``.
    label_fn:
        ``vars (B, k) long -> (B,) long`` task labels.
    k:
        Number of variables.
    v:
        Cardinality per variable.
    n_labels:
        Number of task labels.
    """

    def __init__(
        self,
        gt_variables_fn: Callable[[torch.Tensor], torch.Tensor],
        label_fn: Callable[[torch.Tensor], torch.Tensor],
        k: int,
        v: int = 2,
        n_labels: int = 2,
    ) -> None:
        super().__init__()
        if k <= 0 or v <= 1 or n_labels <= 0:
            raise CausalModelError(f"invalid k={k}, v={v}, n_labels={n_labels}")
        self._gt_variables_fn = gt_variables_fn
        self._label_fn = label_fn
        self.k_max = k
        self.v = v
        self.n_labels = n_labels

    def set_temperature(self, tau: float) -> None:  # noqa: D401 - interface parity
        """No-op: a fixed model has no temperature."""

    def variable_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ground-truth integer variable values ``(B, k)`` long."""
        vals = self._gt_variables_fn(inputs).to(torch.long)
        if vals.shape[1] != self.k_max:
            raise CausalModelError(
                f"gt_variables_fn returned k={vals.shape[1]}, expected {self.k_max}"
            )
        return vals

    def variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """One-hot variables ``(B, k, v)`` (hard; no gradient)."""
        vals = self.variable_values(inputs)
        return torch.nn.functional.one_hot(vals, num_classes=self.v).to(torch.float32)

    def _labels_to_logits(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert integer labels ``(B,)`` to hard one-hot logits ``(B, n_labels)``."""
        return torch.nn.functional.one_hot(
            labels.to(torch.long), num_classes=self.n_labels
        ).to(torch.float32)

    def predict(self, inputs: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Clean label logits ``(B, n_labels)`` (one-hot of ``label_fn``)."""
        _reject_gate(gate)
        labels = self._label_fn(self.variable_values(inputs))
        return self._labels_to_logits(labels)

    def counterfactual_predict(
        self,
        base_inputs: torch.Tensor,
        source_inputs: torch.Tensor,
        source_assignment: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Counterfactual label logits ``(B, n_labels)`` (hard, no gradient).

        Same swap semantics as :meth:`LearnedCausalModel.counterfactual_predict`
        but operating on integer variable values.  Gating is a learned-model
        concept (the fixed hypothesis keeps all its variables), so a non-``None``
        ``gate`` is rejected.
        """
        _reject_gate(gate)
        b = base_inputs.shape[0]
        m = source_inputs.shape[1]
        _check_assignment(source_assignment, b, self.k_max, m)

        base_vals = self.variable_values(base_inputs)  # (B, k)
        src_flat = source_inputs.reshape(b * m, *source_inputs.shape[2:])
        src_vals = self.variable_values(src_flat).reshape(b, m, self.k_max)

        assign = source_assignment.to(torch.long)
        gather_j = assign.clamp(min=0)
        idx = gather_j.unsqueeze(1)  # (B, 1, k)
        chosen = torch.gather(src_vals, 1, idx).squeeze(1)  # (B, k)
        mixed = torch.where(assign >= 0, chosen, base_vals)
        labels = self._label_fn(mixed)
        return self._labels_to_logits(labels)


def _apply_gate(onehots: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """Mask variable one-hots ``(B, k, v)`` by a straight-through hard gate.

    ``v_used_i = hard(g_i) * v_i``: forward multiplies by ``(g_i > 0.5)`` so a
    gated-off variable's one-hot becomes all-zero (constant-0 in H); the
    straight-through gradient of :func:`jdas.gates.VariableGates.hard` still
    reaches ``log_alpha`` from the value mask.
    """
    if gate is None:
        return onehots
    from .gates import VariableGates

    k = onehots.shape[1]
    if gate.shape != (k,):
        raise CausalModelError(f"gate shape {tuple(gate.shape)} != ({k},)")
    hard_gate = VariableGates.hard(gate.to(onehots.dtype))  # (k,)
    return onehots * hard_gate.view(1, k, 1)


def _reject_gate(gate: torch.Tensor | None) -> None:
    """A fixed causal model cannot be gated (it keeps its exact hypothesis)."""
    if gate is not None:
        raise CausalModelError(
            "FixedCausalModel does not support gates (fixed-H methods keep all "
            "their variables); pass gate=None"
        )


def _check_assignment(assign: torch.Tensor, b: int, k: int, m: int) -> None:
    """Validate a ``source_assignment`` tensor shape and value range."""
    if assign.shape != (b, k):
        raise CausalModelError(
            f"source_assignment shape {tuple(assign.shape)} != ({b}, {k})"
        )
    if assign.numel() and (int(assign.max()) >= m or int(assign.min()) < -1):
        raise CausalModelError(
            f"source_assignment values must be in [-1, {m}), got range "
            f"[{int(assign.min())}, {int(assign.max())}]"
        )
