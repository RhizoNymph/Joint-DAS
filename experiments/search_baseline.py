"""Discrete search baseline for Phase A.

For a (task, site) pair, enumerate every unordered pair of distinct candidate
binary variables from a small hypothesis library, and for each pair build a
:class:`jdas.causal_model.FixedCausalModel` (k=2) whose decoder is fitted as a
*majority-label lookup* over the candidate pair's value combinations.  We then
train the alignment (rotation ``Q`` + subspace layout) with a classic
:class:`jdas.training.DASTrainer` and evaluate held-out ``iia_1`` / ``iia_2``.

Key question: does brute-force discrete search over hand-built hypotheses select
``{E1, E2}`` (or an equivalent basis) and does its best IIA match the
gradient-joint method's ~0.96?

Library (6 candidates -> 15 unordered distinct pairs), from
:func:`jdas.hypotheses.hypothesis_library`:

- ``hierarchical_equality``: ``{E1, E2, XNOR(=y), AND, OR, NAND}``.
- ``boolean_comp``:          ``{A, x3, OR(=y), notA, notx3, XOR}``.

Outputs
-------
- a JSON with per-pair ``clean_task_acc``, ``iia_1``, ``iia_2``,
  ``combined_score`` (mean of iia_1/iia_2), sorted best-first;
- a markdown ranking table.

CLI: ``--task --site-layer --seed --steps --device --out``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import torch

from jdas.causal_model import FixedCausalModel
from jdas.eval import iia
from jdas.hypotheses import hypothesis_library
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import DASTrainer, JointConfig


def _load_task(name: str):
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
    from jdas.models.toy import load_or_train_toy_model

    return load_or_train_toy_model(
        task, site_layer, device, cache_dir="experiments/toy_ckpts"
    )


def _make_layout(d: int, k: int) -> SubspaceLayout:
    return SubspaceLayout(d, k, init_width=max(1.0, d / (2 * k)))


def _candidate_values(task, name: str, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate the hypothesis library on ``inputs`` (returns name -> (N,) long)."""
    atoms = task.gt_variables(inputs).to(torch.long)  # (N, k_gt)
    return hypothesis_library(name, atoms)


def _fit_lookup_decoder(
    v1: torch.Tensor, v2: torch.Tensor, labels: torch.Tensor, v: int
) -> tuple[dict[tuple[int, int], int], int]:
    """Majority-label lookup over ``(v1, v2)`` combos on fit data.

    Returns ``(table, global_majority)`` where ``table[(a, b)]`` is the modal
    label among rows with ``(v1, v2) == (a, b)``, and ``global_majority`` is the
    modal label over all rows (used for combos never seen in fit).
    """
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
    """A k=2 FixedCausalModel for the (cand_a, cand_b) hypothesis pair.

    Variables are ``(cand_a(input), cand_b(input))``; the decoder is the fitted
    majority-label lookup.  Returns ``(model, clean_task_acc)`` where the
    accuracy is of the fitted lookup on the fit set (how well this candidate
    pair can reconstruct the task label at all).
    """
    fit_cands = _candidate_values(task, task_name, fit_inputs)
    v1 = fit_cands[cand_a]
    v2 = fit_cands[cand_b]
    table, global_majority = _fit_lookup_decoder(v1, v2, fit_labels, v)

    # Clean task accuracy of the fitted lookup on the fit set.
    preds = torch.tensor(
        [table.get((int(a), int(b)), global_majority) for a, b in zip(v1, v2, strict=True)]
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


def _render_markdown(res: dict) -> str:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Discrete search baseline (Phase A)")
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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    task = _load_task(args.task)
    site = _load_site(task, args.site_layer, args.device)
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

    # Fit data for the lookup decoders.
    fit_gen = torch.Generator(device=args.device).manual_seed(args.seed + 99)
    fit_inputs, fit_labels = task.sample_inputs(args.fit_samples, fit_gen)
    fit_inputs = fit_inputs.to(args.device)
    fit_labels = fit_labels.to(args.device)

    # Candidate names come from the hypothesis library.
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
        layout = _make_layout(d, model.k_max)
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
    md_path.write_text(_render_markdown(res))
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
