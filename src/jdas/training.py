"""Trainers and configs for Joint-DAS.

- :class:`JointTrainer` -- learns the rotation ``Q``, subspace boundaries, and
  the causal model ``H`` jointly (the proposed method).
- :class:`DASTrainer` -- classic DAS: ``H`` fixed, only ``Q`` and boundaries
  train (baseline / freeze-and-refit).
- :func:`refit_rotation` -- freeze a learned ``H`` (as a
  :class:`FixedCausalModel`) and refit ``Q`` alone; the "freeze-and-refit"
  protocol.

Loss structure (N frozen throughout)
------------------------------------
- ``L_cf`` -- cross-entropy between N's intervened log-probs and H's
  counterfactual straight-through one-hot target (so gradients reach ``H``
  through the straight-through estimator and reach ``Q``/boundaries through the
  intervention), plus a symmetric term training H's encoders toward N's hard
  intervened label (weight 0.5).
- ``L_task`` -- CE of ``H.predict`` on clean base and source inputs vs. true
  labels; forces H to implement the task.
- ``L_sparse`` -- ``lambda_sparse * total_aligned_dims / d``.

Temperatures (Gumbel/straight-through ``tau_g`` and mask ``tau_m``) are annealed
from ``*_start`` to ``*_end`` over training (linear or cosine).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .causal_model import FixedCausalModel, LearnedCausalModel
from .eval import effective_k, iia, iia_live
from .gates import VariableGates
from .intervention import interchange
from .rotation import OrthogonalRotation, SubspaceLayout
from .types import InterventionBatch, InterventionSite, Task


class TrainingError(ValueError):
    """Raised for invalid training configuration."""


@dataclass
class JointConfig:
    """Configuration for :class:`JointTrainer` / :class:`DASTrainer`.

    Attributes cover the optimization loop, loss weights, temperature schedules,
    evaluation cadence, and reproducibility.
    """

    steps: int = 2000
    batch_size: int = 64
    n_sources: int = 2
    lr: float = 1e-3
    lr_rotation: float | None = None  # defaults to lr
    lr_causal: float | None = None  # defaults to lr
    lambda_task: float = 1.0
    lambda_sparse: float = 0.1
    # Per-variable hard-concrete (L0) gates for minimality.  ``use_gates``
    # enables the gate machinery (a VariableGates module must be supplied to the
    # trainer); ``lambda_gate`` weights the L0 penalty.  ``use_gates=True`` with
    # ``lambda_gate=0`` is the parameterization control (gates present but
    # costless).
    use_gates: bool = False
    lambda_gate: float = 0.0
    # Gate optimization knobs.  ``gate_init`` is the initial ``log_alpha`` for
    # every gate (default ``+2.0`` ~= 0.88 open).  ``gate_lr`` is a dedicated
    # learning rate for the gate parameters (``None`` -> use ``lr``): Adam moves
    # ``log_alpha`` by at most ~lr per step, so at ``lr=1e-3`` the init ``+2.0``
    # cannot reach the ``-0.4`` needed to close a gate within a few hundred to a
    # couple thousand steps; a larger ``gate_lr`` (e.g. ``0.05``) lets the L0
    # penalty actually close gates.
    gate_lr: float | None = None
    gate_init: float = 2.0
    # Gate training schedule (RESULTS.md N3.3: the gate system is bistable —
    # whichever gradient dominates while gates are still mobile wins the race,
    # and the L0 penalty is never the deciding force).  These three knobs
    # control the schedule so pruning is deliberate rather than a race artifact.
    #   gate_warmup_steps: while step < this, training is numerically identical
    #     to a no-gates run — gate=None threaded everywhere, no gate penalty, and
    #     gate params receive no updates.  Lets variables become causally useful
    #     before any pruning pressure applies.
    #   gate_lambda_ramp_steps: after warmup, the effective lambda_gate scales
    #     linearly 0 -> lambda_gate over this many steps (0 = instant full λ).
    #   gate_clamp: after each active optimizer step, clamp gates.log_alpha to
    #     [-gate_clamp, +gate_clamp] so neither the open- nor closed-saturation
    #     region kills the penalty/sample gradient.  None disables the clamp.
    gate_warmup_steps: int = 0
    gate_lambda_ramp_steps: int = 0
    gate_clamp: float | None = 3.0
    # Sparsity penalty parameterization:
    #   "normalized" -> L_sparse = total_aligned_dims / d  (per-dim grad lambda/d)
    #   "per_dim"    -> L_sparse = total_aligned_dims       (per-dim grad lambda)
    # At LM scale (d=1536) "normalized" is too weak; "per_dim" gives a per-dim
    # gradient of lambda itself so the penalty bites regardless of d.
    sparse_mode: str = "normalized"
    lambda_cf_symmetric: float = 0.5
    # Temperature schedules.
    gumbel_temp_start: float = 1.0
    gumbel_temp_end: float = 0.1
    mask_temp_start: float = 1.0
    mask_temp_end: float = 0.1
    anneal: str = "linear"  # "linear" | "cosine"
    # Swap-size distribution over |I|.
    swap_sizes: tuple[int, ...] = (1, 2)
    swap_weights: tuple[float, ...] = (0.5, 0.5)
    # Evaluation.
    eval_every: int = 200
    eval_batches: int = 4
    eval_batch_size: int = 64
    # Repro / device.
    seed: int = 0
    device: str = "cpu"
    # Control: freeze Q at random init.
    freeze_rotation: bool = False

    def resolved_lr_rotation(self) -> float:
        return self.lr if self.lr_rotation is None else self.lr_rotation

    def resolved_lr_causal(self) -> float:
        return self.lr if self.lr_causal is None else self.lr_causal

    def resolved_gate_lr(self) -> float:
        return self.lr if self.gate_lr is None else self.gate_lr


def _anneal(start: float, end: float, step: int, total: int, mode: str) -> float:
    """Interpolate ``start -> end`` at ``step`` of ``total`` (linear/cosine)."""
    if total <= 1:
        return end
    frac = min(max(step / (total - 1), 0.0), 1.0)
    match mode:
        case "linear":
            return start + (end - start) * frac
        case "cosine":
            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))
        case _:
            raise TrainingError(f"unknown anneal mode {mode!r}")


def _sample_swap_assignment(
    b: int,
    k_max: int,
    n_sources: int,
    swap_sizes: tuple[int, ...],
    swap_weights: tuple[float, ...],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample a ``(b, k_max)`` assignment; per row pick a swap size then swap
    that many distinct variables each from a distinct source.
    """
    if len(swap_sizes) != len(swap_weights):
        raise TrainingError("swap_sizes and swap_weights must have equal length")
    weights = torch.tensor(swap_weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    assign = torch.full((b, k_max), -1, dtype=torch.long, device=device)
    sizes = torch.tensor(swap_sizes, device=device)
    picks = torch.multinomial(
        weights.expand(b, -1), 1, replacement=True, generator=generator
    ).squeeze(1)
    for row in range(b):
        s = int(sizes[picks[row]])
        s = min(s, k_max, max(n_sources, 1))
        var_perm = torch.randperm(k_max, generator=generator, device=device)[:s]
        src_perm = torch.randperm(max(n_sources, 1), generator=generator, device=device)[:s]
        assign[row, var_perm] = src_perm
    return assign


def _to_device(batch: InterventionBatch, device: torch.device) -> InterventionBatch:
    return InterventionBatch(
        base_inputs=batch.base_inputs.to(device),
        source_inputs=batch.source_inputs.to(device),
        source_assignment=batch.source_assignment.to(device),
        base_labels=batch.base_labels.to(device),
        source_labels=batch.source_labels.to(device),
    )


def _replace_assignment(batch: InterventionBatch, assign: torch.Tensor) -> InterventionBatch:
    return InterventionBatch(
        base_inputs=batch.base_inputs,
        source_inputs=batch.source_inputs,
        source_assignment=assign,
        base_labels=batch.base_labels,
        source_labels=batch.source_labels,
    )


class _BaseTrainer:
    """Shared loop for joint and classic-DAS training."""

    def __init__(
        self,
        site: InterventionSite,
        task: Task,
        causal_model: nn.Module,
        rotation: OrthogonalRotation,
        layout: SubspaceLayout,
        config: JointConfig,
        *,
        train_causal: bool,
        gates: VariableGates | None = None,
    ) -> None:
        self.site = site
        self.task = task
        self.causal_model = causal_model
        self.rotation = rotation
        self.layout = layout
        self.config = config
        self.train_causal = train_causal
        self.device = torch.device(config.device)

        if config.use_gates:
            if gates is None:
                raise TrainingError(
                    "config.use_gates=True but no VariableGates supplied to the "
                    "trainer"
                )
            if gates.k_max != layout.k_max:
                raise TrainingError(
                    f"gates.k_max {gates.k_max} != layout.k_max {layout.k_max}"
                )
            if not train_causal:
                raise TrainingError(
                    "gates apply only to learned methods (train_causal=True); "
                    "fixed-H baselines keep their exact hypothesis"
                )
        elif gates is not None:
            raise TrainingError(
                "gates supplied but config.use_gates=False; set use_gates=True to "
                "enable them"
            )
        self.gates = gates

        self.rotation.to(self.device)
        self.layout.to(self.device)
        if isinstance(self.causal_model, nn.Module):
            self.causal_model.to(self.device)
        if self.gates is not None:
            self.gates.to(self.device)

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.eval_generator = torch.Generator(device=self.device)
        self.eval_generator.manual_seed(config.seed + 10_000)

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups: list[dict] = []
        rot_params = [p for p in self.rotation.parameters() if p.requires_grad]
        if rot_params:
            groups.append({"params": rot_params, "lr": self.config.resolved_lr_rotation()})
        layout_params = [p for p in self.layout.parameters() if p.requires_grad]
        if layout_params:
            groups.append({"params": layout_params, "lr": self.config.lr})
        if self.train_causal and isinstance(self.causal_model, nn.Module):
            cm_params = [p for p in self.causal_model.parameters() if p.requires_grad]
            if cm_params:
                groups.append({"params": cm_params, "lr": self.config.resolved_lr_causal()})
        if self.gates is not None:
            gate_params = [p for p in self.gates.parameters() if p.requires_grad]
            if gate_params:
                groups.append(
                    {"params": gate_params, "lr": self.config.resolved_gate_lr()}
                )
        if not groups:
            raise TrainingError("no trainable parameters for the optimizer")
        return torch.optim.AdamW(groups)

    def _set_temperatures(self, step: int) -> None:
        cfg = self.config
        tau_g = _anneal(cfg.gumbel_temp_start, cfg.gumbel_temp_end, step, cfg.steps, cfg.anneal)
        tau_m = _anneal(cfg.mask_temp_start, cfg.mask_temp_end, step, cfg.steps, cfg.anneal)
        self.layout.set_temperature(tau_m)
        self.causal_model.set_temperature(tau_g)

    def _gates_active(self, step: int) -> bool:
        """True when gates participate in the forward/loss/update at ``step``.

        During warmup (``step < gate_warmup_steps``) gates are inert: gate=None
        is threaded everywhere, no penalty is added, and gate params get no
        updates — a warmup step is numerically identical to a no-gates run.
        """
        return self.gates is not None and step >= self.config.gate_warmup_steps

    def _gate_phase(self, step: int) -> str:
        """Schedule phase at ``step``: ``warmup`` | ``ramp`` | ``active``."""
        cfg = self.config
        if step < cfg.gate_warmup_steps:
            return "warmup"
        if step < cfg.gate_warmup_steps + cfg.gate_lambda_ramp_steps:
            return "ramp"
        return "active"

    def _effective_lambda_gate(self, step: int) -> float:
        """Scheduled λ_gate at ``step``.

        0 during warmup; linear 0 -> ``config.lambda_gate`` across the ramp;
        full ``config.lambda_gate`` afterwards.
        """
        cfg = self.config
        if step < cfg.gate_warmup_steps:
            return 0.0
        ramp = cfg.gate_lambda_ramp_steps
        if ramp <= 0:
            return cfg.lambda_gate
        frac = (step - cfg.gate_warmup_steps) / ramp
        frac = min(max(frac, 0.0), 1.0)
        return cfg.lambda_gate * frac

    def _compute_losses(
        self, batch: InterventionBatch, step: int = 0
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        # CRITICAL INVARIANT: sample the gate ONCE here and pass the SAME sample
        # to both N's interchange (width scaling) and H's counterfactual/task
        # predictions (value mask), so a dead variable is a no-op on BOTH sides
        # within this one forward/loss computation.  During warmup gates are
        # inert (gate=None), making the step identical to a no-gates run.
        gate = None
        if self._gates_active(step):
            gate = self.gates.sample(generator=self.generator)

        # Interchange logits from the frozen network N (soft masks, grads flow).
        n_logits = interchange(
            self.site, self.rotation, self.layout, batch, hard=False, gate=gate
        )
        n_logprobs = F.log_softmax(n_logits, dim=-1)

        # H's counterfactual prediction.
        h_cf_logits = self.causal_model.counterfactual_predict(
            batch.base_inputs, batch.source_inputs, batch.source_assignment, gate=gate
        )
        # Straight-through hard one-hot target for L_cf (grads reach H through ST).
        h_cf_target = _st_onehot_from_logits(h_cf_logits)  # (B, n_labels)
        l_cf = -(h_cf_target * n_logprobs).sum(dim=-1).mean()

        losses = {"l_cf": l_cf}

        # Symmetric term: train H's encoders toward N's hard intervened label.
        if self.train_causal:
            n_hard = n_logits.argmax(-1).detach()
            l_cf_sym = F.cross_entropy(h_cf_logits, n_hard)
            losses["l_cf_sym"] = l_cf_sym
        else:
            losses["l_cf_sym"] = n_logits.new_zeros(())

        # L_task: H implements the task on clean base + source inputs.  Uses the
        # same gate so a dead variable is inert for the task fit too.
        if self.train_causal:
            base_pred = self.causal_model.predict(batch.base_inputs, gate=gate)
            l_task = F.cross_entropy(base_pred, batch.base_labels)
            b, m = batch.source_inputs.shape[0], batch.source_inputs.shape[1]
            src_flat = batch.source_inputs.reshape(b * m, *batch.source_inputs.shape[2:])
            src_pred = self.causal_model.predict(src_flat, gate=gate)
            l_task = l_task + F.cross_entropy(src_pred, batch.source_labels.reshape(b * m))
            losses["l_task"] = l_task
        else:
            losses["l_task"] = n_logits.new_zeros(())

        # L_sparse on total aligned dims.  "normalized" divides by d (per-dim
        # gradient lambda/d); "per_dim" leaves it unnormalized (per-dim gradient
        # lambda), which bites at large d.
        total_aligned = self.layout.total_aligned_dims(gate=gate)
        match cfg.sparse_mode:
            case "normalized":
                l_sparse = total_aligned / self.layout.d
            case "per_dim":
                l_sparse = total_aligned
            case _:
                raise TrainingError(f"unknown sparse_mode {cfg.sparse_mode!r}")
        losses["l_sparse"] = l_sparse

        total = (
            l_cf
            + cfg.lambda_cf_symmetric * losses["l_cf_sym"]
            + cfg.lambda_task * losses["l_task"]
            + cfg.lambda_sparse * l_sparse
        )

        # L_gate: expected number of open gates (L0 minimality).  Only added to
        # the loss when gates are *active* (past warmup) and with the *scheduled*
        # effective λ (0 during warmup, ramped 0->lambda_gate, then full).
        # use_gates + lambda_gate=0 remains the costless parameterization
        # control.  During warmup the penalty is omitted entirely so the step is
        # identical to a no-gates run.
        if self._gates_active(step):
            l_gate = self.gates.penalty()
            losses["l_gate"] = l_gate
            total = total + self._effective_lambda_gate(step) * l_gate

        losses["total"] = total
        return losses

    def _post_step(self, step: int) -> None:
        """Post-optimizer-step gate maintenance (clamp log_alpha in place).

        Runs only when gates are active (past warmup) and ``gate_clamp`` is set,
        so neither the open- nor closed-saturation region kills the penalty /
        sample gradient.  A no-op during warmup — combined with ``gate=None`` in
        the forward (so ``log_alpha.grad`` stays ``None`` and AdamW skips the
        param), this guarantees gate params are untouched during warmup.
        """
        if not self._gates_active(step):
            return
        clamp = self.config.gate_clamp
        if clamp is None:
            return
        with torch.no_grad():
            self.gates.log_alpha.clamp_(-clamp, clamp)

    def _sample_training_batch(self) -> InterventionBatch:
        cfg = self.config
        batch = self.task.sample_batch(
            cfg.batch_size, cfg.n_sources, self.layout.k_max, self.generator
        )
        batch = _to_device(batch, self.device)
        assign = _sample_swap_assignment(
            batch.base_inputs.shape[0],
            self.layout.k_max,
            cfg.n_sources,
            cfg.swap_sizes,
            cfg.swap_weights,
            self.generator,
            self.device,
        )
        return _replace_assignment(batch, assign)

    def _evaluate(self) -> dict[str, object]:
        cfg = self.config
        iia_scores = iia(
            self.site,
            self.rotation,
            self.layout,
            self.causal_model,
            self.task,
            n_batches=cfg.eval_batches,
            batch_size=cfg.eval_batch_size,
            n_sources=cfg.n_sources,
            generator=self.eval_generator,
            swap_sizes=(1, 2),
        )
        eff_k = effective_k(
            self.site,
            self.rotation,
            self.layout,
            self.task,
            n_batches=cfg.eval_batches,
            batch_size=cfg.eval_batch_size,
            n_sources=cfg.n_sources,
            generator=self.eval_generator,
        )
        record: dict[str, object] = {
            # None (JSON null) when a swap size is inapplicable (e.g. k_max=1
            # models have no |I|=2 interventions) — 0.0 would fake a failure.
            "iia_1": iia_scores.get(1),
            "iia_2": iia_scores.get(2),
            "effective_k": eff_k,
            "aligned_dims": float(self.layout.total_aligned_dims().item()),
            "hard_widths": self.layout.hard_widths().tolist(),
        }
        if self.gates is not None:
            g_det = self.gates.deterministic()
            live_scores = iia_live(
                self.site,
                self.rotation,
                self.layout,
                self.causal_model,
                self.task,
                self.gates,
                n_batches=cfg.eval_batches,
                batch_size=cfg.eval_batch_size,
                n_sources=cfg.n_sources,
                generator=self.eval_generator,
                swap_sizes=(1, 2),
            )
            record.update(
                {
                    "gated_k": self.gates.gated_k(),
                    "g_det": [round(float(x), 4) for x in g_det.tolist()],
                    "live_indices": self.gates.live_indices(),
                    # gate-scaled hard widths (live vars' actual subspace widths).
                    "hard_widths_gated": self.layout.hard_widths(gate=g_det).tolist(),
                    "iia_1_live": live_scores.get(1),
                    "iia_2_live": live_scores.get(2),
                }
            )
        return record

    def train(self) -> dict[str, object]:
        """Run the training loop; return ``{"history": [...], "final": {...}}``."""
        cfg = self.config
        history: list[dict[str, object]] = []
        for step in range(cfg.steps):
            self._set_temperatures(step)
            batch = self._sample_training_batch()
            losses = self._compute_losses(batch, step)

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()
            self._post_step(step)

            if cfg.eval_every > 0 and (step % cfg.eval_every == 0 or step == cfg.steps - 1):
                record: dict[str, object] = {
                    "step": step,
                    "loss_total": float(losses["total"].item()),
                    "loss_cf": float(losses["l_cf"].item()),
                    "loss_cf_sym": float(losses["l_cf_sym"].item()),
                    "loss_task": float(losses["l_task"].item()),
                    "loss_sparse": float(losses["l_sparse"].item()),
                    "tau_gumbel": self._current_tau_g(step),
                    "tau_mask": self.layout.temperature,
                }
                if self.gates is not None:
                    # Make the schedule dynamics visible in result JSONs.
                    record["gate_phase"] = self._gate_phase(step)
                    record["lambda_gate_eff"] = self._effective_lambda_gate(step)
                if "l_gate" in losses:
                    record["loss_gate"] = float(losses["l_gate"].item())
                with torch.no_grad():
                    record.update(self._evaluate())
                history.append(record)

        final = self._evaluate()
        return {"history": history, "final": final}

    def _current_tau_g(self, step: int) -> float:
        cfg = self.config
        return _anneal(
            cfg.gumbel_temp_start, cfg.gumbel_temp_end, step, cfg.steps, cfg.anneal
        )


class JointTrainer(_BaseTrainer):
    """Joint training of ``Q``, subspace boundaries, and the causal model ``H``."""

    def __init__(
        self,
        site: InterventionSite,
        task: Task,
        causal_model: LearnedCausalModel,
        rotation: OrthogonalRotation,
        layout: SubspaceLayout,
        config: JointConfig,
        gates: VariableGates | None = None,
    ) -> None:
        super().__init__(
            site, task, causal_model, rotation, layout, config,
            train_causal=True, gates=gates,
        )


class DASTrainer(_BaseTrainer):
    """Classic DAS: fixed causal model, only ``Q`` and boundaries train."""

    def __init__(
        self,
        site: InterventionSite,
        task: Task,
        causal_model: nn.Module,
        rotation: OrthogonalRotation,
        layout: SubspaceLayout,
        config: JointConfig,
    ) -> None:
        super().__init__(
            site, task, causal_model, rotation, layout, config, train_causal=False
        )


def _st_onehot_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Straight-through one-hot of ``argmax(logits)`` over the last dim.

    Forward = one-hot(argmax); backward = softmax gradient.  Used to build a
    hard target that still passes gradients into ``H``'s counterfactual logits.
    """
    soft = torch.softmax(logits, dim=-1)
    idx = torch.argmax(logits, dim=-1)
    hard = F.one_hot(idx, num_classes=logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


def refit_rotation(
    site: InterventionSite,
    task: Task,
    learned_model: LearnedCausalModel,
    config: JointConfig,
    *,
    fresh_rotation: bool = True,
    gates: VariableGates | None = None,
) -> dict[str, object]:
    """Freeze-and-refit: freeze the learned ``H`` (hard argmax) and refit ``Q``.

    Wrap ``learned_model``'s hard argmax behavior as a
    :class:`FixedCausalModel`, then run a :class:`DASTrainer` on a fresh (or
    continued) rotation + layout.  Returns the DAS training result plus the
    final IIA.

    Parameters
    ----------
    fresh_rotation:
        If ``True`` reinitialize ``Q`` and the subspace layout; else the caller
        should pass in already-initialized modules via a fresh call.
    """
    device = torch.device(config.device)
    k = learned_model.k_max
    v = learned_model.v

    # Freeze-and-refit uses a FixedCausalModel (no gates).  When the joint run
    # used gates, bake the discovered liveness into the frozen H: a dead
    # variable's discretized value is forced to constant 0, so it stays inert
    # under refit.  Refit itself runs plain DAS (use_gates off).
    hard_gate = None
    if gates is not None:
        hard_gate = gates.live_mask().to(device).long()  # (k,) 1=live, 0=dead

    @torch.no_grad()
    def gt_vars(inputs: torch.Tensor) -> torch.Tensor:
        vals = learned_model.variables(inputs.to(device)).argmax(-1)  # (B, k)
        if hard_gate is not None:
            vals = vals * hard_gate.view(1, -1)
        return vals

    @torch.no_grad()
    def label_fn(vals: torch.Tensor) -> torch.Tensor:
        onehots = F.one_hot(vals.to(torch.long), num_classes=v).to(torch.float32)
        return learned_model.decode(onehots).argmax(-1)

    frozen = FixedCausalModel(gt_vars, label_fn, k=k, v=v, n_labels=learned_model.n_labels)

    # Refit runs plain DAS on the frozen (already-discretized) H; gates are a
    # training-only mechanism and must be disabled for the fixed-H refit.
    refit_config = replace(config, use_gates=False, lambda_gate=0.0)
    d = site.d
    rotation = OrthogonalRotation(d, freeze=refit_config.freeze_rotation)
    layout = SubspaceLayout(d, k, init_width=max(1.0, d / (2 * k)))
    trainer = DASTrainer(site, task, frozen, rotation, layout, refit_config)
    result = trainer.train()
    result["refit_iia_1"] = result["final"]["iia_1"]
    result["refit_iia_2"] = result["final"]["iia_2"]
    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class CheckpointError(ValueError):
    """Raised for invalid checkpoint save/load configuration."""


def _layout_meta(layout: SubspaceLayout) -> dict:
    return {
        "d": layout.d,
        "k_max": layout.k_max,
        "max_width": layout.max_width,
        "min_temp": layout.min_temp,
        "max_temp": layout.max_temp,
    }


def _causal_meta(causal_model: nn.Module) -> dict:
    """JSON-serializable hyperparameters needed to rebuild a learned model.

    Only :class:`LearnedCausalModel` (and its subclasses, e.g.
    ``FeaturizedCausalModel``) carry trainable state worth restoring; fixed
    models are rebuilt from task callables by the caller and need no state.
    """
    if not isinstance(causal_model, LearnedCausalModel):
        return {"kind": "fixed"}
    decoder = causal_model.decoder
    # Decoder is either nn.Linear (decoder_hidden=None) or a 2-layer Sequential.
    if isinstance(decoder, nn.Linear):
        decoder_hidden = None
    else:
        # Sequential(Linear(k*v, h), ReLU, Linear(h, n_labels)); read h.
        decoder_hidden = int(decoder[0].out_features)
    encoder_hidden = int(causal_model.encoders[0][0].out_features)
    return {
        "kind": "learned",
        "input_dim": causal_model.input_dim,
        "k_max": causal_model.k_max,
        "v": causal_model.v,
        "n_labels": causal_model.n_labels,
        "encoder_hidden": encoder_hidden,
        "decoder_hidden": decoder_hidden,
    }


def _gate_meta(gates: VariableGates | None, config: JointConfig) -> dict:
    """JSON-serializable hyperparameters needed to rebuild a gates module.

    ``gate_init``/``gate_lr`` record the optimization knobs used for this run
    (``log_alpha`` itself is restored from the gate state dict, so ``gate_init``
    is informational for reconstruction/reproducibility).
    """
    if gates is None:
        return {"present": False}
    return {
        "present": True,
        "k_max": gates.k_max,
        "beta": gates.beta,
        "gamma": gates.gamma,
        "zeta": gates.zeta,
        "gate_init": config.gate_init,
        "gate_lr": config.gate_lr,
        # Schedule knobs (informational; the schedule is a training-time control,
        # log_alpha is restored from the state dict).  Older gated checkpoints
        # predate these and default gracefully on load.
        "gate_warmup_steps": config.gate_warmup_steps,
        "gate_lambda_ramp_steps": config.gate_lambda_ramp_steps,
        "gate_clamp": config.gate_clamp,
    }


def save_checkpoint(
    path: str | Path,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    causal_model: nn.Module,
    config: JointConfig,
    extra: dict | None = None,
    gates: VariableGates | None = None,
) -> None:
    """Save rotation + layout + (learned) causal-model state and reconstruction meta.

    Persists ``state_dict``s plus a JSON-serializable ``meta`` block carrying the
    hyperparameters needed to rebuild each module (site dim ``d``, ``k_max``,
    ``v``, decoder/encoder widths, ``max_width``/``init_width`` semantics).  A
    :class:`FixedCausalModel` has no learnable state, so only its ``kind`` is
    recorded and the caller must supply the callables at load time.

    Parameters
    ----------
    path:
        Destination file (``torch.save`` format).
    rotation, layout, causal_model:
        The trained modules.
    config:
        Training config (stored via :func:`dataclasses.asdict`).
    extra:
        Optional JSON-serializable dict merged into ``meta["extra"]`` (e.g.
        final metrics, tags).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "d": rotation.d,
        "rotation_frozen": rotation.frozen,
        "layout": _layout_meta(layout),
        "causal": _causal_meta(causal_model),
        "gates": _gate_meta(gates, config),
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "extra": extra or {},
    }
    # Validate JSON-serializability early (fail before writing the blob).
    json.dumps(meta)
    payload = {
        "meta": meta,
        "rotation_state": rotation.state_dict(),
        "layout_state": layout.state_dict(),
        "causal_state": (
            causal_model.state_dict()
            if isinstance(causal_model, LearnedCausalModel)
            else None
        ),
        "gates_state": gates.state_dict() if gates is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    feature_fn=None,
    map_location: str | torch.device = "cpu",
    expect_gates: bool | None = None,
) -> dict:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Reconstructs the :class:`OrthogonalRotation` and :class:`SubspaceLayout` from
    ``meta`` and loads their state.  If the checkpoint held a learned causal
    model it is rebuilt too: a plain :class:`LearnedCausalModel` when
    ``feature_fn is None``, otherwise a
    :class:`jdas.models.hf.FeaturizedCausalModel` wrapping ``feature_fn``.  A
    :class:`jdas.gates.VariableGates` is rebuilt and its ``log_alpha`` restored
    when the checkpoint carried gates.

    Parameters
    ----------
    expect_gates:
        If ``None`` (default) the loader accepts whatever the checkpoint holds.
        If ``True`` the checkpoint *must* carry gates (else
        :class:`CheckpointError`); if ``False`` it must *not* (loading a
        mismatched gate/no-gate config silently would be a correctness bug).

    Returns
    -------
    dict
        ``{"rotation", "layout", "causal_model", "gates", "config", "meta"}``.
        ``causal_model`` is ``None`` for checkpoints whose model was fixed;
        ``gates`` is ``None`` for checkpoints saved without gates.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    meta = payload["meta"]

    # Older checkpoints predate the gates field; treat them as gate-less.
    gate_meta = meta.get("gates", {"present": False})
    has_gates = bool(gate_meta.get("present", False))
    if expect_gates is True and not has_gates:
        raise CheckpointError(
            "expect_gates=True but the checkpoint was saved without gates"
        )
    if expect_gates is False and has_gates:
        raise CheckpointError(
            "expect_gates=False but the checkpoint was saved with gates"
        )

    lm = meta["layout"]
    layout = SubspaceLayout(
        lm["d"],
        lm["k_max"],
        max_width=lm["max_width"],
        min_temp=lm["min_temp"],
        max_temp=lm["max_temp"],
    )
    layout.load_state_dict(payload["layout_state"])
    layout.to(map_location)

    rotation = OrthogonalRotation(meta["d"], freeze=meta["rotation_frozen"])
    rotation.load_state_dict(payload["rotation_state"])
    rotation.to(map_location)

    causal_model = None
    cm = meta["causal"]
    if cm["kind"] == "learned":
        if feature_fn is None:
            causal_model = LearnedCausalModel(
                cm["input_dim"],
                cm["k_max"],
                v=cm["v"],
                n_labels=cm["n_labels"],
                encoder_hidden=cm["encoder_hidden"],
                decoder_hidden=cm["decoder_hidden"],
            )
        else:
            from .models.hf import FeaturizedCausalModel

            causal_model = FeaturizedCausalModel(
                feature_fn,
                cm["input_dim"],
                cm["k_max"],
                v=cm["v"],
                n_labels=cm["n_labels"],
                encoder_hidden=cm["encoder_hidden"],
                decoder_hidden=cm["decoder_hidden"],
            )
        causal_model.load_state_dict(payload["causal_state"])
        causal_model.to(map_location)

    gates = None
    if has_gates:
        # Older gated checkpoints predate gate_init/gate_lr; default them.  The
        # log_alpha values are restored from the state dict below, so init only
        # affects the pre-load parameter values (immediately overwritten).
        gates = VariableGates(
            gate_meta["k_max"],
            init=gate_meta.get("gate_init", 2.0),
            beta=gate_meta["beta"],
            gamma=gate_meta["gamma"],
            zeta=gate_meta["zeta"],
        )
        gates.load_state_dict(payload["gates_state"])
        gates.to(map_location)

    return {
        "rotation": rotation,
        "layout": layout,
        "causal_model": causal_model,
        "gates": gates,
        "config": meta["config"],
        "meta": meta,
    }
