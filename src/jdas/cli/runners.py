"""Single-run experiment logic for the ``jdas run`` subcommands.

Moved verbatim from the ``experiments/run_phase_a.py`` / ``run_phase_b.py`` /
``search_baseline.py`` / ``seed_study.py`` scripts so there is one home for the
run logic.  Argument names and semantics are unchanged: the thin
``experiments/*`` shims re-export these builders/functions, and the committed
result ``config`` blocks (which serialize ``vars(args)``) stay meaningful.

Each ``build_*_parser`` returns an argparse parser with ``prog`` overridable so
it slots under the ``jdas run <sub>`` command tree without changing flags, and
each ``run_*`` takes an already-parsed ``argparse.Namespace``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path

import torch

from jdas.causal_model import FixedCausalModel, LearnedCausalModel
from jdas.eval import iia, recovery
from jdas.gates import VariableGates
from jdas.hypotheses import hypothesis_library
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import (
    DASTrainer,
    JointConfig,
    JointTrainer,
    refit_rotation,
    save_checkpoint,
)

_GATE_METHODS = ("joint", "random_rotation")


def _validate_gate_method(method: str, gates: bool) -> None:
    """Gates apply only to the learned methods; reject fixed-H combos."""
    if gates and method not in _GATE_METHODS:
        raise SystemExit(
            f"--gates only applies to methods {_GATE_METHODS}; got {method!r} "
            "(fixed-H baselines keep their exact hypothesis)"
        )


# ===========================================================================
# Phase A
# ===========================================================================


def _load_task_a(name: str):
    """Lazily import and construct a task by short name."""
    match name:
        case "hierarchical_equality":
            from jdas.tasks.hierarchical_equality import HierarchicalEqualityTask

            return HierarchicalEqualityTask()
        case "boolean_comp":
            from jdas.tasks.boolean_comp import BooleanCompositionTask

            return BooleanCompositionTask()
        case _:
            raise SystemExit(f"unknown task {name!r}")


def _load_site_a(task, site_layer: int, device: str):
    """Lazily import the toy-model loader and build an intervention site."""
    from jdas.models.toy import load_or_train_toy_model

    return load_or_train_toy_model(
        task, site_layer, device, cache_dir="experiments/toy_ckpts"
    )


def _build_config_a(args: argparse.Namespace) -> JointConfig:
    return JointConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_sources=args.n_sources,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        freeze_rotation=(args.method == "random_rotation"),
        eval_every=max(1, args.steps // 10),
        use_gates=args.gates,
        lambda_gate=args.lambda_gate,
        gate_lr=args.gate_lr,
        gate_init=args.gate_init,
        gate_warmup_steps=args.gate_warmup,
        gate_lambda_ramp_steps=args.gate_lambda_ramp,
        gate_clamp=args.gate_clamp,
    )


def build_phase_a_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Joint-DAS Phase A runner")
    parser.add_argument(
        "--task",
        choices=["hierarchical_equality", "boolean_comp"],
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=["joint", "das_true", "das_wrong", "das_wrong_and", "random_rotation"],
        required=True,
    )
    parser.add_argument("--site-layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-sources", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--v", type=int, default=2)
    parser.add_argument(
        "--gates",
        action="store_true",
        help="enable per-variable hard-concrete (L0) gates (joint/random_rotation only)",
    )
    parser.add_argument(
        "--lambda-gate",
        type=float,
        default=0.0,
        help="weight of the L0 gate penalty (0 = parameterization control)",
    )
    parser.add_argument(
        "--gate-lr",
        type=float,
        default=None,
        help="dedicated learning rate for gate params (None = use --lr)",
    )
    parser.add_argument(
        "--gate-init",
        type=float,
        default=2.0,
        help="initial log_alpha for every gate (+2.0 ~= 0.88 open)",
    )
    parser.add_argument(
        "--gate-warmup",
        type=int,
        default=0,
        help="steps of gate warmup (gates inert = no-gates run) before pruning",
    )
    parser.add_argument(
        "--gate-lambda-ramp",
        type=int,
        default=0,
        help="steps to ramp effective lambda_gate 0->lambda_gate after warmup",
    )
    parser.add_argument(
        "--gate-clamp",
        type=float,
        default=3.0,
        help="clamp log_alpha to [-c, +c] after active steps (keeps grad alive)",
    )
    parser.add_argument("--out", type=str, default="results.json")
    return parser


def run_phase_a(args: argparse.Namespace) -> None:
    _validate_gate_method(args.method, args.gates)

    torch.manual_seed(args.seed)
    task = _load_task_a(args.task)
    site = _load_site_a(task, args.site_layer, args.device)
    config = _build_config_a(args)

    d = site.d
    input_dim = _infer_input_dim(task, args)
    rotation = OrthogonalRotation(d, freeze=config.freeze_rotation)

    result: dict[str, object] = {"config": {**vars(args)}, "config_dataclass": asdict(config)}

    match args.method:
        case "joint" | "random_rotation":
            causal_model = LearnedCausalModel(
                input_dim=input_dim, k_max=args.k_max, v=args.v, n_labels=task.n_labels
            )
            layout = _make_layout(d, causal_model.k_max)
            gates = (
                VariableGates(causal_model.k_max, init=args.gate_init)
                if args.gates
                else None
            )
            trainer = JointTrainer(
                site, task, causal_model, rotation, layout, config, gates=gates
            )
            train_out = trainer.train()
            _add_recovery(result, causal_model, task, config, gates=gates)
            # Freeze-and-refit only makes sense when Q is trainable.
            if args.method == "joint":
                refit = refit_rotation(site, task, causal_model, config, gates=gates)
                result["refit_iia_1"] = refit["refit_iia_1"]
                result["refit_iia_2"] = refit["refit_iia_2"]
        case "das_true":
            causal_model = _true_fixed_model(task, args)
            layout = _make_layout(d, causal_model.k_max)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
        case "das_wrong":
            causal_model = _wrong_fixed_model(task, args)
            layout = _make_layout(d, causal_model.k_max)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
        case "das_wrong_and":
            causal_model = _wrong_and_fixed_model(task, args)
            layout = _make_layout(d, causal_model.k_max)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            # The analytic agreement ceiling: the IIA a *perfect* das run with
            # this wrong law would approach if N is faithful to the true law.
            ceiling = _wrong_and_agreement_ceiling(task, args)
            result["agreement_ceiling"] = ceiling
        case _:
            raise SystemExit(f"unknown method {args.method!r}")

    result["history"] = train_out["history"]
    result["final"] = train_out["final"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


def _make_layout(d: int, k: int) -> SubspaceLayout:
    """Layout sized to the causal model's variable count."""
    return SubspaceLayout(d, k, init_width=max(1.0, d / (2 * k)))


def _infer_input_dim(task, args: argparse.Namespace) -> int:
    """Infer the flattened input dim by sampling one batch."""
    gen = torch.Generator().manual_seed(0)
    batch = task.sample_batch(2, args.n_sources, args.k_max, gen)
    return int(batch.base_inputs.reshape(2, -1).shape[1])


def _add_recovery(result: dict, causal_model, task, config: JointConfig, gates=None) -> None:
    if task.k_gt > 0:
        gen = torch.Generator(device=config.device).manual_seed(config.seed + 1)
        live = gates.live_indices() if gates is not None else None
        rec = recovery(causal_model, task, generator=gen, live_indices=live)
        result["recovery_matrix"] = rec.matrix
        result["best_assignment"] = rec.best_assignment
        result["recovery_score"] = rec.best_score


def _true_fixed_model(task, args: argparse.Namespace) -> FixedCausalModel:
    """FixedCausalModel using the task's ground-truth variables + label rule."""
    label_fn = getattr(task, "label_from_variables", None) or getattr(
        task, "gt_label_fn", None
    )
    if label_fn is None:
        raise SystemExit("task exposes neither label_from_variables nor gt_label_fn")
    return FixedCausalModel(
        gt_variables_fn=task.gt_variables,
        label_fn=label_fn,
        k=task.k_gt,
        v=args.v,
        n_labels=task.n_labels,
    )


def _wrong_fixed_model(task, args: argparse.Namespace) -> FixedCausalModel:
    """Deliberately wrong H: a single output-copy variable (Z = y)."""

    def gt_vars(inputs: torch.Tensor) -> torch.Tensor:
        # Wrong hypothesis: a single output-copy variable Z = y (the label).
        vals = task.gt_variables(inputs)
        label_fn = getattr(task, "label_from_variables", None) or getattr(
            task, "gt_label_fn"
        )
        return label_fn(vals).unsqueeze(1)

    def label_fn(vals: torch.Tensor) -> torch.Tensor:
        return vals[:, 0]

    return FixedCausalModel(
        gt_variables_fn=gt_vars,
        label_fn=label_fn,
        k=1,
        v=max(args.v, task.n_labels),
        n_labels=task.n_labels,
    )


def _wrong_law_label_fn(task_name: str):
    """A deliberately WRONG composition law over the two true GT atoms.

    - ``hierarchical_equality``: label = AND(E1, E2)   (truth is XNOR).
    - ``boolean_comp``:          label = XOR(A, x3)    (truth is OR).

    Both are k=2 laws over the *correct* variables, so the model admits genuine
    |I|=2 multi-source interventions; its counterfactual labels disagree with a
    faithful network at a computable rate (see the agreement ceiling).
    """
    match task_name:
        case "hierarchical_equality":
            return lambda vals: (vals[:, 0] & vals[:, 1]).to(torch.long)
        case "boolean_comp":
            return lambda vals: (vals[:, 0] ^ vals[:, 1]).to(torch.long)
        case _:
            raise SystemExit(f"no wrong-and law for task {task_name!r}")


def _wrong_and_fixed_model(task, args: argparse.Namespace) -> FixedCausalModel:
    """k=2 FixedCausalModel: TRUE GT variables, WRONG composition law."""
    return FixedCausalModel(
        gt_variables_fn=task.gt_variables,
        label_fn=_wrong_law_label_fn(args.task),
        k=task.k_gt,
        v=args.v,
        n_labels=task.n_labels,
    )


def _wrong_and_agreement_ceiling(
    task, args: argparse.Namespace, n_samples: int = 20_000
) -> dict:
    """Analytic agreement ceiling of the wrong-law model vs the true law.

    Over the task's own sampled intervention distribution, compute — *per swap
    size* — the fraction of interventions whose wrong-law counterfactual label
    equals the true-law counterfactual label.  This is the IIA a *perfect* DAS
    run with the wrong law would approach **if N is faithful to the true law**.
    """
    true_label_fn = getattr(task, "label_from_variables", None) or getattr(
        task, "gt_label_fn"
    )
    wrong_label_fn = _wrong_law_label_fn(args.task)
    gen = torch.Generator().manual_seed(args.seed + 4242)
    k = task.k_gt

    ceilings: dict[str, float] = {}
    for swap_size in (1, 2):
        if swap_size > k:
            continue
        batch = task.sample_batch(n_samples, args.n_sources, k, gen)
        base_vals = task.gt_variables(batch.base_inputs)  # (N, k)
        m = batch.source_inputs.shape[1]
        src_flat = batch.source_inputs.reshape(n_samples * m, *batch.source_inputs.shape[2:])
        src_vals = task.gt_variables(src_flat).reshape(n_samples, m, k)
        assign = _fixed_swap_assignment(n_samples, k, args.n_sources, swap_size, gen)
        gather_j = assign.clamp(min=0)
        chosen = torch.gather(src_vals, 1, gather_j.unsqueeze(1)).squeeze(1)  # (N, k)
        mixed = torch.where(assign >= 0, chosen, base_vals)
        true_cf = true_label_fn(mixed)
        wrong_cf = wrong_label_fn(mixed)
        agree = (true_cf == wrong_cf).float().mean().item()
        ceilings[str(swap_size)] = round(agree, 4)
    return ceilings


def _fixed_swap_assignment(
    b: int, k: int, n_sources: int, swap_size: int, generator: torch.Generator
) -> torch.Tensor:
    """``(b, k)`` assignment swapping exactly ``swap_size`` distinct vars, each
    from a distinct source (mirrors :func:`jdas.eval._build_assignment`)."""
    assign = torch.full((b, k), -1, dtype=torch.long)
    for row in range(b):
        var_perm = torch.randperm(k, generator=generator)[:swap_size]
        src_perm = torch.randperm(n_sources, generator=generator)[:swap_size]
        assign[row, var_perm] = src_perm
    return assign


# ===========================================================================
# Phase B
# ===========================================================================


def _build_config_b(args: argparse.Namespace) -> JointConfig:
    return JointConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_sources=args.n_sources,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        freeze_rotation=(args.method == "random_rotation"),
        eval_every=max(1, args.steps // 10),
        lambda_sparse=args.lambda_sparse,
        sparse_mode=args.sparse_mode,
        use_gates=args.gates,
        lambda_gate=args.lambda_gate,
        gate_lr=args.gate_lr,
        gate_init=args.gate_init,
        gate_warmup_steps=args.gate_warmup,
        gate_lambda_ramp_steps=args.gate_lambda_ramp,
        gate_clamp=args.gate_clamp,
    )


def _true_fixed_model_b(task, v: int) -> FixedCausalModel:
    """FixedCausalModel with the true GT variables + AND label rule."""
    return FixedCausalModel(
        gt_variables_fn=task.gt_variables,
        label_fn=task.label_from_variables,
        k=task.k_gt,
        v=v,
        n_labels=task.n_labels,
    )


def _wrong_fixed_model_b(task, v: int, k: int) -> FixedCausalModel:
    """Deliberately wrong H: a single output-copy variable (Z = y).

    The variable count is padded to ``k`` (the layout's ``k_max``) with dead,
    always-zero variables so that ``|I|=2`` interchange evaluations remain valid.
    """
    v_eff = max(v, task.n_labels)

    def gt_vars(inputs: torch.Tensor) -> torch.Tensor:
        vals = task.gt_variables(inputs)
        y = task.label_from_variables(vals)  # (B,) output copy
        out = torch.zeros(y.shape[0], k, dtype=torch.long, device=y.device)
        out[:, 0] = y
        return out

    def label_fn(vals: torch.Tensor) -> torch.Tensor:
        return vals[:, 0]

    return FixedCausalModel(
        gt_variables_fn=gt_vars,
        label_fn=label_fn,
        k=k,
        v=v_eff,
        n_labels=task.n_labels,
    )


def _maybe_save_ckpt_b(
    args: argparse.Namespace,
    rotation,
    layout,
    causal_model,
    config: JointConfig,
    train_out: dict,
    gates: VariableGates | None = None,
) -> None:
    """Save a checkpoint (rotation+layout+causal state + meta) if ``--save-ckpt``."""
    if not args.save_ckpt:
        return
    save_checkpoint(
        args.save_ckpt,
        rotation,
        layout,
        causal_model,
        config,
        extra={
            "method": args.method,
            "layer": args.layer,
            "model": args.model,
            "init_width": args.init_width,
            "final": train_out.get("final", {}),
        },
        gates=gates,
    )
    print(f"saved checkpoint to {args.save_ckpt}")


def _add_recovery_b(result: dict, causal_model, task, config: JointConfig, gates=None) -> None:
    if task.k_gt > 0:
        gen = torch.Generator(device=config.device).manual_seed(config.seed + 1)
        live = gates.live_indices() if gates is not None else None
        rec = recovery(causal_model, task, generator=gen, live_indices=live)
        result["recovery_matrix"] = rec.matrix
        result["best_assignment"] = rec.best_assignment
        result["recovery_score"] = rec.best_score


def build_phase_b_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Joint-DAS Phase B runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=["joint", "das_true", "das_wrong", "random_rotation"],
        required=True,
    )
    parser.add_argument("--template-id", type=int, default=0)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="render prompts through the tokenizer chat template",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-sources", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--v", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--lambda-sparse",
        type=float,
        default=0.1,
        help="weight of the aligned-dims sparsity penalty (0.1 is too weak at LM scale)",
    )
    parser.add_argument(
        "--sparse-mode",
        choices=["normalized", "per_dim"],
        default="normalized",
        help="normalized: lambda*total/d (weak at large d); per_dim: lambda*total",
    )
    parser.add_argument(
        "--gates",
        action="store_true",
        help="enable per-variable hard-concrete (L0) gates (joint/random_rotation only)",
    )
    parser.add_argument(
        "--lambda-gate",
        type=float,
        default=0.0,
        help="weight of the L0 gate penalty (0 = parameterization control)",
    )
    parser.add_argument(
        "--gate-lr",
        type=float,
        default=None,
        help="dedicated learning rate for gate params (None = use --lr)",
    )
    parser.add_argument(
        "--gate-init",
        type=float,
        default=2.0,
        help="initial log_alpha for every gate (+2.0 ~= 0.88 open)",
    )
    parser.add_argument(
        "--gate-warmup",
        type=int,
        default=0,
        help="steps of gate warmup (gates inert = no-gates run) before pruning",
    )
    parser.add_argument(
        "--gate-lambda-ramp",
        type=int,
        default=0,
        help="steps to ramp effective lambda_gate 0->lambda_gate after warmup",
    )
    parser.add_argument(
        "--gate-clamp",
        type=float,
        default=3.0,
        help="clamp log_alpha to [-c, +c] after active steps (keeps grad alive)",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=None,
        help="hard per-variable width cap (dims); None = unbounded softplus widths",
    )
    parser.add_argument(
        "--init-width",
        type=float,
        default=None,
        help="initial per-variable width (dims); default d/(2*k_max)",
    )
    parser.add_argument(
        "--position",
        choices=["last", "z_digits"],
        default="last",
        help="intervention position: last token, or last token of the Z amount",
    )
    parser.add_argument(
        "--save-ckpt",
        type=str,
        default=None,
        help="path to save a checkpoint (rotation+layout+causal) after training",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="skip the freeze-and-refit pass after joint training (halves runtime)",
    )
    parser.add_argument("--out", type=str, default="results.json")
    return parser


def run_phase_b(args: argparse.Namespace) -> None:
    from jdas.models.hf import FeaturizedCausalModel, load_hf_site
    from jdas.tasks.price_tagging import PriceTaggingTask

    _validate_gate_method(args.method, args.gates)

    torch.manual_seed(args.seed)

    site = load_hf_site(
        args.model, args.layer, args.device, local_files_only=args.local_files_only
    )
    task = PriceTaggingTask(
        site.tokenizer,
        template_id=args.template_id,
        device=args.device,
        use_chat_template=args.chat_template,
        position=args.position,
    )
    config = _build_config_b(args)

    d = site.d
    match args.method:
        case "joint" | "random_rotation":
            k_max = args.k_max
        case "das_true":
            k_max = task.k_gt
        case "das_wrong":
            k_max = args.k_max
        case _:
            raise SystemExit(f"unknown method {args.method!r}")
    rotation = OrthogonalRotation(d, freeze=config.freeze_rotation)
    init_width = args.init_width if args.init_width is not None else max(1.0, d / (2 * k_max))
    if args.max_width is not None:
        init_width = min(init_width, 0.5 * args.max_width)
    layout = SubspaceLayout(d, k_max, init_width=init_width, max_width=args.max_width)

    result: dict[str, object] = {
        "config": {**vars(args)},
        "config_dataclass": asdict(config),
    }

    match args.method:
        case "joint" | "random_rotation":
            causal_model = FeaturizedCausalModel(
                feature_fn=task.causal_features,
                input_dim=task.input_dim,
                k_max=k_max,
                v=args.v,
                n_labels=task.n_labels,
            )
            gates = VariableGates(k_max, init=args.gate_init) if args.gates else None
            trainer = JointTrainer(
                site, task, causal_model, rotation, layout, config, gates=gates
            )
            train_out = trainer.train()
            _maybe_save_ckpt_b(
                args, rotation, layout, causal_model, config, train_out, gates=gates
            )
            _add_recovery_b(result, causal_model, task, config, gates=gates)
            if args.method == "joint" and not args.no_refit:
                refit = refit_rotation(site, task, causal_model, config, gates=gates)
                result["refit_iia_1"] = refit["refit_iia_1"]
                result["refit_iia_2"] = refit["refit_iia_2"]
        case "das_true":
            causal_model = _true_fixed_model_b(task, args.v)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _maybe_save_ckpt_b(args, rotation, layout, causal_model, config, train_out)
        case "das_wrong":
            causal_model = _wrong_fixed_model_b(task, args.v, k_max)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _maybe_save_ckpt_b(args, rotation, layout, causal_model, config, train_out)
        case _:
            raise SystemExit(f"unknown method {args.method!r}")

    result["history"] = train_out["history"]
    result["final"] = train_out["final"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


# ===========================================================================
# Search baseline
# ===========================================================================


def _make_layout_search(d: int, k: int) -> SubspaceLayout:
    return SubspaceLayout(d, k, init_width=max(1.0, d / (2 * k)))


def _candidate_values(task, name: str, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    atoms = task.gt_variables(inputs).to(torch.long)  # (N, k_gt)
    return hypothesis_library(name, atoms)


def _fit_lookup_decoder(
    v1: torch.Tensor, v2: torch.Tensor, labels: torch.Tensor, v: int
) -> tuple[dict[tuple[int, int], int], int]:
    table: dict[tuple[int, int], int] = {}
    for a, b in product(range(v), repeat=2):
        sel = (v1 == a) & (v2 == b)
        if int(sel.sum()) == 0:
            continue
        table[(a, b)] = int(torch.mode(labels[sel]).values)
    global_majority = int(torch.mode(labels).values)
    return table, global_majority


def _build_pair_model(
    task,
    task_name: str,
    cand_a: str,
    cand_b: str,
    v: int,
    n_labels: int,
    fit_inputs: torch.Tensor,
    fit_labels: torch.Tensor,
) -> tuple[FixedCausalModel, float]:
    fit_cands = _candidate_values(task, task_name, fit_inputs)
    v1 = fit_cands[cand_a]
    v2 = fit_cands[cand_b]
    table, global_majority = _fit_lookup_decoder(v1, v2, fit_labels, v)

    preds = torch.tensor(
        [table.get((int(a), int(b)), global_majority) for a, b in zip(v1, v2, strict=True)],
        device=fit_labels.device,
    )
    clean_acc = float((preds == fit_labels).float().mean())

    def gt_vars(inputs: torch.Tensor) -> torch.Tensor:
        cands = _candidate_values(task, task_name, inputs)
        return torch.stack([cands[cand_a], cands[cand_b]], dim=-1)

    def label_fn(vals: torch.Tensor) -> torch.Tensor:
        out = torch.empty(vals.shape[0], dtype=torch.long, device=vals.device)
        for i in range(vals.shape[0]):
            key = (int(vals[i, 0]), int(vals[i, 1]))
            out[i] = table.get(key, global_majority)
        return out

    model = FixedCausalModel(
        gt_variables_fn=gt_vars, label_fn=label_fn, k=2, v=v, n_labels=n_labels
    )
    return model, clean_acc


def _render_search_markdown(res: dict) -> str:
    cfg = res["config"]
    lines = [
        f"# Discrete search baseline: {cfg['task']} (layer {cfg['site_layer']})\n",
        f"Config: seed={cfg['seed']} steps={cfg['steps']} device={cfg['device']} "
        f"candidates={res['candidates']}\n",
        "Ranking by combined score (mean of iia_1, iia_2), best first.\n",
        "| rank | V1 | V2 | clean_task_acc | iia_1 | iia_2 | combined |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(res["ranking"], start=1):
        i2 = "-" if r["iia_2"] is None else f"{r['iia_2']:.4f}"
        lines.append(
            f"| {rank} | {r['V1']} | {r['V2']} | {r['clean_task_acc']:.4f} | "
            f"{r['iia_1']:.4f} | {i2} | {r['combined_score']:.4f} |"
        )
    best = res["ranking"][0]
    lines.append(
        f"\nBest pair: **({best['V1']}, {best['V2']})** with combined "
        f"{best['combined_score']:.4f} (iia_1={best['iia_1']:.4f}).\n"
    )
    return "\n".join(lines)


def build_search_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Discrete search baseline (Phase A)")
    p.add_argument(
        "--task", choices=["hierarchical_equality", "boolean_comp"], required=True
    )
    p.add_argument("--site-layer", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-sources", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--v", type=int, default=2)
    p.add_argument("--fit-samples", type=int, default=8000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="experiments/results/search_baseline.json")
    return p


def run_search(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    task = _load_task_a(args.task)
    site = _load_site_a(task, args.site_layer, args.device)
    d = site.d

    config = JointConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_sources=args.n_sources,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        freeze_rotation=False,
        eval_every=max(1, args.steps // 5),
    )

    fit_gen = torch.Generator(device=args.device).manual_seed(args.seed + 99)
    fit_inputs, fit_labels = task.sample_inputs(args.fit_samples, fit_gen)
    fit_inputs = fit_inputs.to(args.device)
    fit_labels = fit_labels.to(args.device)

    dummy_atoms = task.gt_variables(fit_inputs[:2]).to(torch.long)
    candidates = list(hypothesis_library(args.task, dummy_atoms).keys())
    pairs = list(combinations(candidates, 2))  # 15 unordered distinct pairs

    ranking: list[dict] = []
    for cand_a, cand_b in pairs:
        model, clean_acc = _build_pair_model(
            task,
            args.task,
            cand_a,
            cand_b,
            args.v,
            task.n_labels,
            fit_inputs,
            fit_labels,
        )
        rotation = OrthogonalRotation(d, freeze=False)
        layout = _make_layout_search(d, model.k_max)
        trainer = DASTrainer(site, task, model, rotation, layout, config)
        trainer.train()

        eval_gen = torch.Generator(device=args.device).manual_seed(args.seed + 7)
        scores = iia(
            site,
            rotation,
            layout,
            model,
            task,
            n_batches=8,
            batch_size=64,
            n_sources=args.n_sources,
            generator=eval_gen,
            swap_sizes=(1, 2),
        )
        iia_1 = scores.get(1)
        iia_2 = scores.get(2)
        parts = [x for x in (iia_1, iia_2) if x is not None]
        combined = sum(parts) / len(parts) if parts else 0.0
        ranking.append(
            {
                "V1": cand_a,
                "V2": cand_b,
                "clean_task_acc": round(clean_acc, 4),
                "iia_1": None if iia_1 is None else round(iia_1, 4),
                "iia_2": None if iia_2 is None else round(iia_2, 4),
                "combined_score": round(combined, 4),
            }
        )
        print(
            f"({cand_a}, {cand_b}) clean={clean_acc:.3f} "
            f"iia_1={iia_1} iia_2={iia_2} combined={combined:.3f}"
        )

    ranking.sort(key=lambda r: r["combined_score"], reverse=True)
    res = {
        "config": {
            "task": args.task,
            "site_layer": args.site_layer,
            "seed": args.seed,
            "steps": args.steps,
            "device": args.device,
            "fit_samples": args.fit_samples,
        },
        "candidates": candidates,
        "ranking": ranking,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_search_markdown(res))
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")


# ===========================================================================
# Seed study
# ===========================================================================

LIVE_THRESH = 0.02


def build_seed_study_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Seed / basis variance study (Phase A)")
    p.add_argument(
        "--task", choices=["hierarchical_equality", "boolean_comp"],
        default="hierarchical_equality",
    )
    p.add_argument("--site-layer", type=int, default=1)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-sources", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--v", type=int, default=2)
    p.add_argument("--n-probe", type=int, default=4096)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="experiments/results/seed_study.json")
    return p


def run_seed_study(args: argparse.Namespace) -> None:
    # Reuse the shared wiring + per-variable liveness measurement from the
    # introspection script (same code path as run_phase_a).
    import math
    import sys
    from pathlib import Path as _Path

    # The installed ``jdas`` console script does not have the repo root on
    # sys.path, so ``experiments.*`` (a non-installed namespace package) needs it.
    _repo_root = str(_Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from experiments.introspect_phase_a import (
        _infer_input_dim as _isp_infer_input_dim,
        _load_site as _isp_load_site,
        _load_task as _isp_load_task,
        _make_layout as _isp_make_layout,
        _per_variable_effect,
    )
    from jdas.hypotheses import best_matching_fn, classify_solution

    task = _isp_load_task(args.task)
    site = _isp_load_site(task, args.site_layer, args.device)

    def _train_joint(seed: int):
        config = JointConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            n_sources=args.n_sources,
            lr=args.lr,
            seed=seed,
            device=args.device,
            freeze_rotation=False,
            eval_every=max(1, args.steps // 10),
        )
        torch.manual_seed(seed)
        d = site.d
        input_dim = _isp_infer_input_dim(task, args.k_max, args.n_sources)
        rotation = OrthogonalRotation(d, freeze=False)
        causal_model = LearnedCausalModel(
            input_dim=input_dim, k_max=args.k_max, v=args.v, n_labels=task.n_labels
        )
        layout = _isp_make_layout(d, causal_model.k_max)
        trainer = JointTrainer(site, task, causal_model, rotation, layout, config)
        train_out = trainer.train()
        return causal_model, rotation, layout, config, train_out["final"]

    @torch.no_grad()
    def _classify_seed(causal_model, rotation, layout, config, seed: int):
        device = torch.device(args.device)
        causal_model.eval()
        effect_gen = torch.Generator(device=device).manual_seed(seed + 999)
        effect_rates = _per_variable_effect(
            site, rotation, layout, task,
            n_batches=8, batch_size=256, generator=effect_gen,
        )
        widths = layout.hard_widths().tolist()
        live_idx = [
            i for i in range(causal_model.k_max)
            if widths[i] >= 1 and effect_rates[i] > LIVE_THRESH
        ]
        gen = torch.Generator(device=device).manual_seed(seed + 777)
        probe = task.sample_batch(args.n_probe, args.n_sources, causal_model.k_max, gen)
        inputs = probe.base_inputs.to(device)
        atoms = task.gt_variables(inputs).to(device).to(torch.long)  # (N, 2)
        learned_argmax = causal_model.variables(inputs).argmax(-1)  # (N, k_max)
        live_values = [learned_argmax[:, i] for i in live_idx]
        classification = classify_solution(live_values, atoms, task_name=args.task)
        live_fns = []
        for i, vals in zip(live_idx, live_values, strict=True):
            name, acc = best_matching_fn(vals, atoms)
            live_fns.append(
                {"var": i, "effect_rate": round(effect_rates[i], 4),
                 "fn": name, "fn_agreement": round(acc, 4)}
            )
        rec_gen = torch.Generator(device=device).manual_seed(seed + 1)
        rec = recovery(causal_model, task, generator=rec_gen)
        return {
            "classification": classification,
            "live_idx": live_idx,
            "live_fns": live_fns,
            "effect_rates": [round(x, 4) for x in effect_rates],
            "recovery_score": round(rec.best_score, 4),
        }

    def _mean_std(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return 0.0, 0.0
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return mean, math.sqrt(var)

    per_seed: list[dict] = []
    for seed in range(args.seeds):
        causal_model, rotation, layout, config, final = _train_joint(seed)
        cls = _classify_seed(causal_model, rotation, layout, config, seed)
        row = {
            "seed": seed,
            "classification": cls["classification"],
            "live_idx": cls["live_idx"],
            "live_fns": cls["live_fns"],
            "effect_rates": cls["effect_rates"],
            "iia_1": final["iia_1"],
            "iia_2": final["iia_2"],
            "effective_k": final["effective_k"],
            "hard_widths": final["hard_widths"],
            "recovery_score": cls["recovery_score"],
        }
        per_seed.append(row)
        print(
            f"seed {seed}: class={row['classification']} "
            f"iia_1={row['iia_1']:.4f} eff_k={row['effective_k']} "
            f"live={[f['fn'] for f in row['live_fns']]}"
        )

    classes = ["atoms", "equivalent_basis", "output_copy", "other"]
    class_counts = {c: sum(1 for r in per_seed if r["classification"] == c) for c in classes}
    iia1_mean, iia1_std = _mean_std([r["iia_1"] for r in per_seed])
    iia2_mean, iia2_std = _mean_std([r["iia_2"] for r in per_seed])

    res = {
        "config": {
            "task": args.task,
            "site_layer": args.site_layer,
            "seeds": args.seeds,
            "steps": args.steps,
            "k_max": args.k_max,
            "v": args.v,
            "device": args.device,
            "n_probe": args.n_probe,
        },
        "per_seed": per_seed,
        "aggregate": {
            "class_counts": class_counts,
            "iia_1_mean": round(iia1_mean, 4),
            "iia_1_std": round(iia1_std, 4),
            "iia_2_mean": round(iia2_mean, 4),
            "iia_2_std": round(iia2_std, 4),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_seed_study_markdown(res))
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")


def _render_seed_study_markdown(res: dict) -> str:
    cfg = res["config"]
    lines = [
        f"# Seed / basis variance study: {cfg['task']} (layer {cfg['site_layer']})\n",
        f"Config: seeds={cfg['seeds']} steps={cfg['steps']} k_max={cfg['k_max']} "
        f"v={cfg['v']} device={cfg['device']}\n",
        "## Per-seed classification\n",
        "| seed | class | live vars (fn) | iia_1 | iia_2 | eff_k | recovery |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res["per_seed"]:
        fns = ", ".join(f"Z{f['var']}={f['fn']}" for f in r["live_fns"]) or "(none)"
        i2 = "-" if r["iia_2"] is None else f"{r['iia_2']:.4f}"
        lines.append(
            f"| {r['seed']} | {r['classification']} | {fns} | {r['iia_1']:.4f} | "
            f"{i2} | {r['effective_k']} | {r['recovery_score']:.4f} |"
        )
    agg = res["aggregate"]
    lines.append("\n## Aggregate\n")
    lines.append("Class counts:")
    for cls, n in agg["class_counts"].items():
        lines.append(f"- `{cls}`: {n}")
    lines.append(
        f"\nIIA: iia_1 = {agg['iia_1_mean']:.4f} ± {agg['iia_1_std']:.4f}, "
        f"iia_2 = {agg['iia_2_mean']:.4f} ± {agg['iia_2_std']:.4f} "
        f"(over {cfg['seeds']} seeds).\n"
    )
    return "\n".join(lines)
