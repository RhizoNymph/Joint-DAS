"""Phase A entry point: toy models with known ground-truth causal structure.

Runs one (task, method, site-layer, seed) configuration and writes a JSON
result: config + training history + final metrics.

Methods
-------
- ``joint``          -- learn Q, boundaries, and H jointly (ours).
- ``das_true``       -- classic DAS with the true hand-specified H (upper bound).
- ``das_wrong``      -- classic DAS with a wrong H (single output-copy variable).
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
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import DASTrainer, JointConfig, JointTrainer, refit_rotation


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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint-DAS Phase A runner")
    parser.add_argument(
        "--task",
        choices=["hierarchical_equality", "boolean_comp"],
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=["joint", "das_true", "das_wrong", "random_rotation"],
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
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

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
            trainer = JointTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _add_recovery(result, causal_model, task, config)
            # Freeze-and-refit only makes sense when Q is trainable.
            if args.method == "joint":
                refit = refit_rotation(site, task, causal_model, config)
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


def _add_recovery(result: dict, causal_model, task, config: JointConfig) -> None:
    if task.k_gt > 0:
        gen = torch.Generator(device=config.device).manual_seed(config.seed + 1)
        rec = recovery(causal_model, task, generator=gen)
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


if __name__ == "__main__":
    main()
