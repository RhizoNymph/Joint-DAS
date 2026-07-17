"""Phase B entry point: real HF language model (price-tagging task).

Runs one ``(model, layer, method, template, seed)`` configuration and writes a
JSON result mirroring :mod:`experiments.run_phase_a`: config + training history
+ final metrics (+ recovery / refit for the learned methods).

Methods
-------
- ``joint``           -- learn Q, boundaries, and H jointly (ours).
- ``das_true``        -- classic DAS with the true H (``L=(Z>=X)``, ``U=(Z<=Y)``,
                         AND) -- upper bound / sanity.
- ``das_wrong``       -- classic DAS with a single output-copy variable (Z = y).
- ``random_rotation`` -- joint H but Q frozen at random init (control).

The intervention site is the residual stream at the last token of one decoder
layer of a frozen HF causal LM (:class:`jdas.models.hf.HFSite`).  The learned
causal model reads *decoded* ``(X, Y, Z)`` features from the token ids via
:meth:`PriceTaggingTask.causal_features` (see
:class:`jdas.models.hf.FeaturizedCausalModel`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from jdas.causal_model import FixedCausalModel
from jdas.eval import iia, recovery
from jdas.models.hf import FeaturizedCausalModel, load_hf_site
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.tasks.price_tagging import PriceTaggingTask
from jdas.training import (
    DASTrainer,
    JointConfig,
    JointTrainer,
    refit_rotation,
    save_checkpoint,
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
        lambda_sparse=args.lambda_sparse,
        sparse_mode=args.sparse_mode,
    )


def _true_fixed_model(task: PriceTaggingTask, v: int) -> FixedCausalModel:
    """FixedCausalModel with the true GT variables + AND label rule."""
    return FixedCausalModel(
        gt_variables_fn=task.gt_variables,
        label_fn=task.label_from_variables,
        k=task.k_gt,
        v=v,
        n_labels=task.n_labels,
    )


def _wrong_fixed_model(task: PriceTaggingTask, v: int, k: int) -> FixedCausalModel:
    """Deliberately wrong H: a single output-copy variable (Z = y).

    The variable count is padded to ``k`` (the layout's ``k_max``) with dead,
    always-zero variables so that ``|I|=2`` interchange evaluations remain valid;
    only variable 0 (the output copy) affects the label.
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


def _maybe_save_ckpt(
    args: argparse.Namespace,
    rotation,
    layout,
    causal_model,
    config: JointConfig,
    train_out: dict,
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
    )
    print(f"saved checkpoint to {args.save_ckpt}")


def _add_recovery(result: dict, causal_model, task, config: JointConfig) -> None:
    if task.k_gt > 0:
        gen = torch.Generator(device=config.device).manual_seed(config.seed + 1)
        rec = recovery(causal_model, task, generator=gen)
        result["recovery_matrix"] = rec.matrix
        result["best_assignment"] = rec.best_assignment
        result["recovery_score"] = rec.best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint-DAS Phase B runner")
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
    args = parser.parse_args()

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
    config = _build_config(args)

    d = site.d
    # The layout's k_max must match the causal model's number of variables so
    # the interchange assignment (shape (B, layout.k_max)) is validated against
    # the causal model.  Learned methods use the full k_max; the fixed-H
    # baselines use their model's variable count (k_gt for das_true, 1 for
    # das_wrong).
    match args.method:
        case "joint" | "random_rotation":
            k_max = args.k_max
        case "das_true":
            k_max = task.k_gt
        case "das_wrong":
            # Keep the full k_max so |I|=2 eval works; extra vars are dead.
            k_max = args.k_max
        case _:
            raise SystemExit(f"unknown method {args.method!r}")
    rotation = OrthogonalRotation(d, freeze=config.freeze_rotation)
    init_width = args.init_width if args.init_width is not None else max(1.0, d / (2 * k_max))
    if args.max_width is not None:
        # init must sit strictly below the cap.
        init_width = min(init_width, 0.5 * args.max_width)
    layout = SubspaceLayout(
        d, k_max, init_width=init_width, max_width=args.max_width
    )

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
            trainer = JointTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _maybe_save_ckpt(args, rotation, layout, causal_model, config, train_out)
            _add_recovery(result, causal_model, task, config)
            if args.method == "joint" and not args.no_refit:
                refit = refit_rotation(site, task, causal_model, config)
                result["refit_iia_1"] = refit["refit_iia_1"]
                result["refit_iia_2"] = refit["refit_iia_2"]
        case "das_true":
            causal_model = _true_fixed_model(task, args.v)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _maybe_save_ckpt(args, rotation, layout, causal_model, config, train_out)
        case "das_wrong":
            causal_model = _wrong_fixed_model(task, args.v, k_max)
            trainer = DASTrainer(site, task, causal_model, rotation, layout, config)
            train_out = trainer.train()
            _maybe_save_ckpt(args, rotation, layout, causal_model, config, train_out)
        case _:
            raise SystemExit(f"unknown method {args.method!r}")

    result["history"] = train_out["history"]
    result["final"] = train_out["final"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
