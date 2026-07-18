"""Phase A entry point: toy models with known ground-truth causal structure.

Runs one (task, method, site-layer, seed) configuration and writes a JSON
result: config + training history + final metrics.

Methods
-------
- ``joint``          -- learn Q, boundaries, and H jointly (ours).
- ``das_true``       -- classic DAS with the true hand-specified H (upper bound).
- ``das_wrong``      -- classic DAS with a wrong H (single output-copy variable).
- ``das_wrong_and``  -- classic DAS with the TRUE ground-truth variables but a
  WRONG composition law (k=2), the principled falsification baseline: it admits
  real |I|=2 multi-source interventions whose counterfactual predictions must
  systematically disagree with any faithful network.  The run JSON also records
  the analytic agreement ceiling (see :func:`_wrong_and_agreement_ceiling`).
- ``random_rotation``-- joint H but Q frozen at random init (control).

Tasks/models are imported lazily by module path (they may be authored by a
sibling agent) and are expected to satisfy the ``Task`` / ``InterventionSite``
protocols from :mod:`jdas.types`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from jdas.causal_model import FixedCausalModel, LearnedCausalModel
from jdas.eval import iia, recovery
from jdas.gates import VariableGates
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import DASTrainer, JointConfig, JointTrainer, refit_rotation

_GATE_METHODS = ("joint", "random_rotation")


def _validate_gate_method(method: str, gates: bool) -> None:
    """Gates apply only to the learned methods; reject fixed-H combos."""
    if gates and method not in _GATE_METHODS:
        raise SystemExit(
            f"--gates only applies to methods {_GATE_METHODS}; got {method!r} "
            "(fixed-H baselines keep their exact hypothesis)"
        )


def _load_task(name: str):
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


def _load_site(task, site_layer: int, device: str):
    """Lazily import the toy-model loader and build an intervention site."""
    from jdas.models.toy import load_or_train_toy_model

    return load_or_train_toy_model(
        task, site_layer, device, cache_dir="experiments/toy_ckpts"
    )


def _build_config(args: argparse.Namespace) -> JointConfig:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint-DAS Phase A runner")
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_gate_method(args.method, args.gates)

    torch.manual_seed(args.seed)
    task = _load_task(args.task)
    site = _load_site(task, args.site_layer, args.device)
    config = _build_config(args)

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
    run with the wrong law would approach **if N is faithful to the true law**:

    - measured IIA near this ceiling  => falsification works (the wrong law only
      agrees where the two laws coincide, N follows the truth).
    - measured IIA well above ceiling => something is off (N is not faithful, or
      the alignment is vacuously satisfying the wrong law).

    Everything is computed with the task's ground-truth logic (no network); the
    "network" here is assumed perfectly faithful to the true law.
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


if __name__ == "__main__":
    main()
