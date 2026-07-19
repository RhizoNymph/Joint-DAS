"""Introspect a joint-DAS learned causal model (toy-model scientific analysis).

Retrains a joint run with the *same code path* as ``jdas run toy`` (default:
hierarchical_equality, joint, site-layer 1, seed 0, 4000 steps) and then opens up
the learned causal model H_theta to answer: *what did the learned variables
become?*

For each learned variable ``Z_i`` (``i in 0..k_max-1``), over a large batch of
fresh inputs, we report the best value-relabeling agreement with a set of
candidate hypotheses computed from the ground-truth task variables:

- ``E1 = (a == b)``          (first equality; GT var 0)
- ``E2 = (c == d)``          (second equality; GT var 1)
- ``XOR(E1, E2)``            (== NOT label for hierarchical_equality)
- ``y`` the task label       (== (E1 == E2))
- ``const0`` a constant      (degeneracy check)

We also report, per variable, the *hard mask width* and the *single-variable
causal-effect rate*: the fraction of base inputs for which swapping only ``Z_i``'s
aligned subspace (from a source) flips the frozen network's argmax output. This
is exactly the per-variable liveness test used by :func:`jdas.eval.effective_k`.

Finally we dump the *decoder truth table*: for each combination of values of the
effective (live) variables, the learned decoder's predicted label (other, dead
variables held at their most-common value).

Outputs (written next to the results dir):
- ``experiments/results/introspect_<tag>.json``  (machine-readable)
- ``experiments/results/introspect_<tag>.md``    (human-readable section)

The ``<tag>`` defaults to ``hier_l1_s0`` and can be overridden with ``--tag``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import permutations, product
from pathlib import Path

import torch

from jdas.causal_model import LearnedCausalModel
from jdas.eval import iia, recovery
from jdas.intervention import interchange
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import JointConfig, JointTrainer, refit_rotation
from jdas.types import InterventionBatch


# -- task/site loading (mirrors jdas run toy) ----------------------------------


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


def _infer_input_dim(task, k_max: int, n_sources: int) -> int:
    gen = torch.Generator().manual_seed(0)
    batch = task.sample_batch(2, n_sources, k_max, gen)
    return int(batch.base_inputs.reshape(2, -1).shape[1])


# -- hypothesis construction ---------------------------------------------------


def _hypotheses(task, gt_vars: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Candidate binary hypotheses over the same inputs.

    ``gt_vars`` is ``(N, k_gt)`` long (task ground-truth variables), ``labels`` is
    ``(N,)`` long task labels. Returns a dict name -> ``(N,)`` long tensor.
    """
    e1 = gt_vars[:, 0]
    hyps: dict[str, torch.Tensor] = {"GT0_E1": e1}
    if gt_vars.shape[1] > 1:
        e2 = gt_vars[:, 1]
        hyps["GT1_E2"] = e2
        hyps["XOR(E1,E2)"] = (e1 ^ e2)
    hyps["label_y"] = labels
    hyps["const0"] = torch.zeros_like(e1)
    return hyps


def _best_relabel_agreement(learned: torch.Tensor, target: torch.Tensor, v: int) -> float:
    """Best agreement over value relabelings of ``learned`` (both ``(N,)`` long)."""
    n = learned.shape[0]
    if n == 0:
        return 0.0
    best = 0.0
    for perm in permutations(range(v)):
        perm_t = torch.tensor(perm, device=learned.device)
        acc = (perm_t[learned] == target).float().mean().item()
        best = max(best, acc)
    return best


# -- per-variable causal effect ------------------------------------------------


@torch.no_grad()
def _per_variable_effect(
    site,
    rotation: OrthogonalRotation,
    layout: SubspaceLayout,
    task,
    *,
    n_batches: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[float]:
    """Single-variable causal-effect rate for each variable (hard masks).

    For variable ``i``: swap only ``Z_i``'s aligned subspace from source 0 and
    measure the fraction of examples whose argmax output flips vs. the clean
    forward pass. This mirrors :func:`jdas.eval.effective_k`'s liveness test but
    returns the raw rate per variable.
    """
    device = rotation.matrix.device
    rates: list[float] = []
    for i in range(layout.k_max):
        flipped = 0
        total = 0
        for _ in range(n_batches):
            batch = task.sample_batch(batch_size, 1, layout.k_max, generator)
            batch = _to_device(batch, device)
            b = batch.base_inputs.shape[0]
            assign = torch.full((b, layout.k_max), -1, dtype=torch.long, device=device)
            assign[:, i] = 0
            batch_i = _replace_assignment(batch, assign)
            base_logits = site.logits(batch.base_inputs)
            swapped = interchange(site, rotation, layout, batch_i, hard=True)
            flipped += int((base_logits.argmax(-1) != swapped.argmax(-1)).sum())
            total += b
        rates.append(flipped / max(total, 1))
    return rates


def _to_device(batch: InterventionBatch, device) -> InterventionBatch:
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


# -- decoder truth table -------------------------------------------------------


@torch.no_grad()
def _decoder_truth_table(
    causal_model: LearnedCausalModel,
    effective_idx: list[int],
    default_vals: list[int],
    device,
) -> list[dict]:
    """Decoder predictions over all value combinations of effective variables.

    Dead variables are held at ``default_vals`` (their most-common value). Returns
    a list of ``{"values": {var_i: val}, "pred_label": int, "logits": [...]}``.
    """
    v = causal_model.v
    k = causal_model.k_max
    table: list[dict] = []
    for combo in product(range(v), repeat=len(effective_idx)):
        vals = list(default_vals)
        for slot, idx in enumerate(effective_idx):
            vals[idx] = combo[slot]
        onehot = torch.zeros(1, k, v, device=device)
        for j in range(k):
            onehot[0, j, vals[j]] = 1.0
        logits = causal_model.decode(onehot)
        table.append(
            {
                "values": {f"Z{effective_idx[s]}": combo[s] for s in range(len(effective_idx))},
                "pred_label": int(logits.argmax(-1).item()),
                "logits": [round(float(x), 4) for x in logits.squeeze(0).tolist()],
            }
        )
    return table


# -- joint pair truth table (Z_i, Z_j) vs (E1, E2) -----------------------------


@torch.no_grad()
def _pair_vs_gt_table(
    learned_argmax: torch.Tensor,
    gt_vars: torch.Tensor,
    i: int,
    j: int,
) -> dict:
    """Empirical joint distribution of (Z_i, Z_j) over (E1, E2) combinations.

    For each of the 4 (E1,E2) combinations, report the modal (Z_i, Z_j) pair and
    its purity (fraction of rows in that GT cell that take the modal pair).
    """
    e1 = gt_vars[:, 0]
    e2 = gt_vars[:, 1]
    zi = learned_argmax[:, i]
    zj = learned_argmax[:, j]
    cells = []
    for a, b in product(range(2), repeat=2):
        sel = (e1 == a) & (e2 == b)
        n = int(sel.sum())
        if n == 0:
            cells.append({"E1": a, "E2": b, "n": 0, "modal_Zi_Zj": None, "purity": 0.0})
            continue
        pairs = zi[sel] * 2 + zj[sel]
        modal = int(torch.mode(pairs).values)
        purity = float((pairs == modal).float().mean())
        cells.append(
            {
                "E1": a,
                "E2": b,
                "n": n,
                "modal_Zi_Zj": [modal // 2, modal % 2],
                "purity": round(purity, 4),
            }
        )
    return {"i": i, "j": j, "cells": cells}


# -- markdown rendering --------------------------------------------------------


def _render_markdown(res: dict) -> str:
    cfg = res["config"]
    lines: list[str] = []
    lines.append(f"# Introspection: {res['tag']}\n")
    lines.append(
        f"Config: task=`{cfg['task']}` method=joint site_layer={cfg['site_layer']} "
        f"seed={cfg['seed']} steps={cfg['steps']} k_max={cfg['k_max']} v={cfg['v']} "
        f"device={cfg['device']}\n"
    )
    lines.append(
        f"Final IIA: iia_1={res['final']['iia_1']:.4f}, iia_2={res['final']['iia_2']:.4f}, "
        f"effective_k={res['final']['effective_k']}, "
        f"recovery_score={res.get('recovery_score'):.4f}, "
        f"refit_iia_1={res.get('refit_iia_1'):.4f}, refit_iia_2={res.get('refit_iia_2'):.4f}\n"
    )
    lines.append(f"Hard mask widths: {res['final']['hard_widths']}\n")

    # variable-hypothesis agreement table
    lines.append("\n## Variable-hypothesis agreement (best value relabeling)\n")
    hyp_names = res["hypothesis_names"]
    header = "| Variable | width | effect_rate | " + " | ".join(hyp_names) + " |"
    sep = "|" + "---|" * (3 + len(hyp_names))
    lines.append(header)
    lines.append(sep)
    for row in res["agreement"]:
        cells = " | ".join(f"{row['agreement'][h]:.3f}" for h in hyp_names)
        lines.append(
            f"| Z{row['var']} | {row['width']} | {row['effect_rate']:.3f} | {cells} |"
        )
    lines.append(
        "\n`effect_rate` = fraction of inputs where a single-variable swap of that "
        "variable flips N's output (liveness).\n"
    )

    # per-variable best hypothesis
    lines.append("\n## Per-variable best match\n")
    for row in res["agreement"]:
        best_h = max(row["agreement"], key=lambda h: row["agreement"][h])
        lines.append(
            f"- **Z{row['var']}** (width {row['width']}, effect {row['effect_rate']:.3f}): "
            f"best = `{best_h}` @ {row['agreement'][best_h]:.3f}"
        )

    # decoder truth table
    lines.append("\n## Learned decoder truth table (effective variables)\n")
    lines.append(f"Effective (live) variable indices: {res['effective_idx']}")
    lines.append(f"Dead variables held at default values: {res['default_vals']}\n")
    if res["decoder_truth_table"]:
        eff = res["effective_idx"]
        head = "| " + " | ".join(f"Z{i}" for i in eff) + " | pred_label | logits |"
        lines.append(head)
        lines.append("|" + "---|" * (len(eff) + 2))
        for entry in res["decoder_truth_table"]:
            vals = " | ".join(str(entry["values"][f"Z{i}"]) for i in eff)
            lines.append(f"| {vals} | {entry['pred_label']} | {entry['logits']} |")

    # pair vs GT
    if res.get("pair_vs_gt"):
        pg = res["pair_vs_gt"]
        lines.append(
            f"\n## Joint (Z{pg['i']}, Z{pg['j']}) vs (E1, E2) "
            "[two highest-effect variables]\n"
        )
        lines.append("| E1 | E2 | n | modal (Zi,Zj) | purity |")
        lines.append("|---|---|---|---|---|")
        for c in pg["cells"]:
            lines.append(
                f"| {c['E1']} | {c['E2']} | {c['n']} | {c['modal_Zi_Zj']} | {c['purity']:.3f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(res.get("interpretation", "(see JSON)"))
    lines.append("")
    return "\n".join(lines)


# -- main ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Introspect a joint-DAS learned model")
    p.add_argument("--task", default="hierarchical_equality")
    p.add_argument("--site-layer", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-sources", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--v", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-probe", type=int, default=4096, help="fresh inputs for agreement")
    p.add_argument("--tag", default="hier_l1_s0")
    p.add_argument("--out-dir", default="experiments/results")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    task = _load_task(args.task)
    site = _load_site(task, args.site_layer, args.device)

    config = JointConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_sources=args.n_sources,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        freeze_rotation=False,
        eval_every=max(1, args.steps // 10),
    )

    d = site.d
    input_dim = _infer_input_dim(task, args.k_max, args.n_sources)
    rotation = OrthogonalRotation(d, freeze=False)
    causal_model = LearnedCausalModel(
        input_dim=input_dim, k_max=args.k_max, v=args.v, n_labels=task.n_labels
    )
    layout = _make_layout(d, causal_model.k_max)

    trainer = JointTrainer(site, task, causal_model, rotation, layout, config)
    train_out = trainer.train()

    device = torch.device(args.device)
    causal_model.eval()

    # Fresh probe inputs.
    gen = torch.Generator(device=device).manual_seed(args.seed + 777)
    probe = task.sample_batch(args.n_probe, args.n_sources, causal_model.k_max, gen)
    inputs = probe.base_inputs.to(device)
    gt_vars = task.gt_variables(inputs).to(device).to(torch.long)  # (N, k_gt)
    labels = task.gt_label_fn(gt_vars).to(device).to(torch.long)  # (N,)

    learned_argmax = causal_model.variables(inputs).argmax(-1)  # (N, k_max)

    hyps = _hypotheses(task, gt_vars, labels)
    hyp_names = list(hyps.keys())

    widths = layout.hard_widths().tolist()
    effect_gen = torch.Generator(device=device).manual_seed(args.seed + 999)
    effect_rates = _per_variable_effect(
        site, rotation, layout, task,
        n_batches=8, batch_size=256, generator=effect_gen,
    )

    agreement = []
    for i in range(causal_model.k_max):
        li = learned_argmax[:, i]
        row = {
            "var": i,
            "width": int(widths[i]),
            "effect_rate": round(effect_rates[i], 4),
            "agreement": {
                h: round(_best_relabel_agreement(li, hyps[h], args.v), 4)
                for h in hyp_names
            },
        }
        agreement.append(row)

    # Effective variables: width >= 1 AND effect_rate above liveness threshold.
    live_thresh = 0.02
    effective_idx = [
        i for i in range(causal_model.k_max)
        if widths[i] >= 1 and effect_rates[i] > live_thresh
    ]

    # Default (most-common) value for every variable, for the decoder table.
    default_vals = [int(torch.mode(learned_argmax[:, i]).values) for i in range(causal_model.k_max)]

    decoder_tt = _decoder_truth_table(causal_model, effective_idx, default_vals, device)

    # Two highest-effect variables -> joint vs (E1, E2).
    pair_vs_gt = None
    if task.k_gt >= 2:
        order = sorted(range(causal_model.k_max), key=lambda i: effect_rates[i], reverse=True)
        i, j = order[0], order[1]
        pair_vs_gt = _pair_vs_gt_table(learned_argmax, gt_vars, i, j)

    # Recovery + refit (same as jdas run toy).
    rec_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    rec = recovery(causal_model, task, generator=rec_gen)
    refit = refit_rotation(site, task, causal_model, config)

    # Build an automatic interpretation string.
    interp = _interpret(agreement, hyp_names, effective_idx, pair_vs_gt)

    res = {
        "tag": args.tag,
        "config": {
            "task": args.task,
            "site_layer": args.site_layer,
            "seed": args.seed,
            "steps": args.steps,
            "k_max": args.k_max,
            "v": args.v,
            "device": args.device,
            "n_probe": args.n_probe,
        },
        "config_dataclass": asdict(config),
        "final": train_out["final"],
        "history": train_out["history"],
        "recovery_matrix": rec.matrix,
        "best_assignment": rec.best_assignment,
        "recovery_score": rec.best_score,
        "refit_iia_1": refit["refit_iia_1"],
        "refit_iia_2": refit["refit_iia_2"],
        "hypothesis_names": hyp_names,
        "agreement": agreement,
        "effect_rates": [round(x, 4) for x in effect_rates],
        "hard_widths": widths,
        "effective_idx": effective_idx,
        "default_vals": default_vals,
        "decoder_truth_table": decoder_tt,
        "pair_vs_gt": pair_vs_gt,
        "interpretation": interp,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"introspect_{args.tag}.json"
    md_path = out_dir / f"introspect_{args.tag}.md"
    json_path.write_text(json.dumps(res, indent=2))
    md_path.write_text(_render_markdown(res))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    # Echo the key table to stdout/log.
    print("\n=== variable-hypothesis agreement ===")
    print("var width effect " + " ".join(f"{h:>12}" for h in hyp_names))
    for row in agreement:
        print(
            f"Z{row['var']}  {row['width']:>4} {row['effect_rate']:.3f} "
            + " ".join(f"{row['agreement'][h]:>12.3f}" for h in hyp_names)
        )
    print("\ninterpretation:\n" + interp)


def _interpret(agreement, hyp_names, effective_idx, pair_vs_gt) -> str:
    """Produce a short honest interpretation of the introspection result."""
    parts: list[str] = []
    # Are the two effective variables approximately {E1, E2}?
    best_per_var = {}
    for row in agreement:
        best_h = max(row["agreement"], key=lambda h: row["agreement"][h])
        best_per_var[row["var"]] = (best_h, row["agreement"][best_h])
    eff_best = {i: best_per_var[i] for i in effective_idx}
    e1e2 = {"GT0_E1", "GT1_E2"}
    matched = {h for (h, _) in eff_best.values()} & e1e2
    strong = all(v[1] >= 0.95 for v in eff_best.values()) if eff_best else False
    if matched == e1e2 and strong:
        parts.append(
            "The two effective learned variables map cleanly onto the ground-truth "
            "atoms {E1=(a==b), E2=(c==d)} (best-relabel agreement >= 0.95). Joint DAS "
            "recovers the GT factorization at this site."
        )
    else:
        parts.append(
            "The effective learned variables do NOT both map cleanly onto {E1, E2}. "
            f"Per-variable best matches among effective vars: "
            + ", ".join(f"Z{i}->{h} ({a:.3f})" for i, (h, a) in eff_best.items())
            + ". This suggests an alternative (possibly relabeled/rotated) but valid "
            "factorization rather than the literal GT atoms."
        )
    if pair_vs_gt is not None:
        purities = [c["purity"] for c in pair_vs_gt["cells"] if c["n"] > 0]
        mean_pure = sum(purities) / max(len(purities), 1)
        parts.append(
            f"Joint (Z{pair_vs_gt['i']},Z{pair_vs_gt['j']}) vs (E1,E2): mean cell purity "
            f"{mean_pure:.3f} (1.0 => the top-two variables jointly determine (E1,E2))."
        )
    return " ".join(parts)


if __name__ == "__main__":
    main()
