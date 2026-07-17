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
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .causal_model import FixedCausalModel, LearnedCausalModel
from .eval import effective_k, iia
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
    ) -> None:
        self.site = site
        self.task = task
        self.causal_model = causal_model
        self.rotation = rotation
        self.layout = layout
        self.config = config
        self.train_causal = train_causal
        self.device = torch.device(config.device)

        self.rotation.to(self.device)
        self.layout.to(self.device)
        if isinstance(self.causal_model, nn.Module):
            self.causal_model.to(self.device)

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
        if not groups:
            raise TrainingError("no trainable parameters for the optimizer")
        return torch.optim.AdamW(groups)

    def _set_temperatures(self, step: int) -> None:
        cfg = self.config
        tau_g = _anneal(cfg.gumbel_temp_start, cfg.gumbel_temp_end, step, cfg.steps, cfg.anneal)
        tau_m = _anneal(cfg.mask_temp_start, cfg.mask_temp_end, step, cfg.steps, cfg.anneal)
        self.layout.set_temperature(tau_m)
        self.causal_model.set_temperature(tau_g)

    def _compute_losses(self, batch: InterventionBatch) -> dict[str, torch.Tensor]:
        cfg = self.config
        # Interchange logits from the frozen network N (soft masks, grads flow).
        n_logits = interchange(self.site, self.rotation, self.layout, batch, hard=False)
        n_logprobs = F.log_softmax(n_logits, dim=-1)

        # H's counterfactual prediction.
        h_cf_logits = self.causal_model.counterfactual_predict(
            batch.base_inputs, batch.source_inputs, batch.source_assignment
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

        # L_task: H implements the task on clean base + source inputs.
        if self.train_causal:
            base_pred = self.causal_model.predict(batch.base_inputs)
            l_task = F.cross_entropy(base_pred, batch.base_labels)
            b, m = batch.source_inputs.shape[0], batch.source_inputs.shape[1]
            src_flat = batch.source_inputs.reshape(b * m, *batch.source_inputs.shape[2:])
            src_pred = self.causal_model.predict(src_flat)
            l_task = l_task + F.cross_entropy(src_pred, batch.source_labels.reshape(b * m))
            losses["l_task"] = l_task
        else:
            losses["l_task"] = n_logits.new_zeros(())

        # L_sparse on total aligned dims.  "normalized" divides by d (per-dim
        # gradient lambda/d); "per_dim" leaves it unnormalized (per-dim gradient
        # lambda), which bites at large d.
        total_aligned = self.layout.total_aligned_dims()
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
        losses["total"] = total
        return losses

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
        return {
            # None (JSON null) when a swap size is inapplicable (e.g. k_max=1
            # models have no |I|=2 interventions) — 0.0 would fake a failure.
            "iia_1": iia_scores.get(1),
            "iia_2": iia_scores.get(2),
            "effective_k": eff_k,
            "aligned_dims": float(self.layout.total_aligned_dims().item()),
            "hard_widths": self.layout.hard_widths().tolist(),
        }

    def train(self) -> dict[str, object]:
        """Run the training loop; return ``{"history": [...], "final": {...}}``."""
        cfg = self.config
        history: list[dict[str, object]] = []
        for step in range(cfg.steps):
            self._set_temperatures(step)
            batch = self._sample_training_batch()
            losses = self._compute_losses(batch)

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()

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
    ) -> None:
        super().__init__(
            site, task, causal_model, rotation, layout, config, train_causal=True
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

    @torch.no_grad()
    def gt_vars(inputs: torch.Tensor) -> torch.Tensor:
        return learned_model.variables(inputs.to(device)).argmax(-1)

    @torch.no_grad()
    def label_fn(vals: torch.Tensor) -> torch.Tensor:
        onehots = F.one_hot(vals.to(torch.long), num_classes=v).to(torch.float32)
        return learned_model.decode(onehots).argmax(-1)

    frozen = FixedCausalModel(gt_vars, label_fn, k=k, v=v, n_labels=learned_model.n_labels)

    d = site.d
    rotation = OrthogonalRotation(d, freeze=config.freeze_rotation)
    layout = SubspaceLayout(d, k, init_width=max(1.0, d / (2 * k)))
    trainer = DASTrainer(site, task, frozen, rotation, layout, config)
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


def save_checkpoint(
    path: str | Path,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    causal_model: nn.Module,
    config: JointConfig,
    extra: dict | None = None,
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
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    feature_fn=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Reconstructs the :class:`OrthogonalRotation` and :class:`SubspaceLayout` from
    ``meta`` and loads their state.  If the checkpoint held a learned causal
    model it is rebuilt too: a plain :class:`LearnedCausalModel` when
    ``feature_fn is None``, otherwise a
    :class:`jdas.models.hf.FeaturizedCausalModel` wrapping ``feature_fn``.

    Returns
    -------
    dict
        ``{"rotation", "layout", "causal_model", "config", "meta"}``.
        ``causal_model`` is ``None`` for checkpoints whose model was fixed.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    meta = payload["meta"]

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

    return {
        "rotation": rotation,
        "layout": layout,
        "causal_model": causal_model,
        "config": meta["config"],
        "meta": meta,
    }
