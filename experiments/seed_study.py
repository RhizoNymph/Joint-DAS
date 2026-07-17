"""Seed / basis variance study for Phase A (hierarchical equality).

For each of ``--seeds`` seeds, train a joint model *exactly* as
``run_phase_a.py`` does (same config defaults), then classify the learned
solution WITHOUT retraining:

1. Identify the *live* variables (per-variable causal-effect rate > 2%), reusing
   the per-variable liveness measurement from ``introspect_phase_a.py``.
2. Compute each live variable's value table over ~4096 fresh inputs vs the two
   GT atoms ``(E1, E2)``.
3. Classify the solution with :func:`jdas.hypotheses.classify_solution` into one
   of ``{atoms, equivalent_basis, output_copy, other}``.

Per-seed we record: classification, each live variable's best-matching boolean
function name, ``iia_1``, ``iia_2``, ``effective_k``, ``recovery_score``.
Aggregate: counts per class and mean±std IIA.

CLI: ``--task --site-layer --seeds --steps --device --out``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Ensure the repo root is importable so ``experiments.*`` resolves when this
# file is run directly as ``python experiments/seed_study.py``.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jdas.causal_model import LearnedCausalModel
from jdas.eval import recovery
from jdas.hypotheses import best_matching_fn, classify_solution
from jdas.rotation import OrthogonalRotation
from jdas.training import JointConfig, JointTrainer

# Reuse the shared wiring + per-variable liveness measurement from the
# introspection script (same code path as run_phase_a.py).
from experiments.introspect_phase_a import (
    _infer_input_dim,
    _load_site,
    _load_task,
    _make_layout,
    _per_variable_effect,
)

LIVE_THRESH = 0.02


def _train_joint(task, site, args, seed: int) -> tuple:
    """Train one joint run; return (causal_model, rotation, layout, config, final)."""
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
    input_dim = _infer_input_dim(task, args.k_max, args.n_sources)
    rotation = OrthogonalRotation(d, freeze=False)
    causal_model = LearnedCausalModel(
        input_dim=input_dim, k_max=args.k_max, v=args.v, n_labels=task.n_labels
    )
    layout = _make_layout(d, causal_model.k_max)
    trainer = JointTrainer(site, task, causal_model, rotation, layout, config)
    train_out = trainer.train()
    return causal_model, rotation, layout, config, train_out["final"]


@torch.no_grad()
def _classify_seed(task, site, causal_model, rotation, layout, config, args, seed: int):
    """Classify the learned solution of one trained joint run (no retraining)."""
    device = torch.device(args.device)
    causal_model.eval()

    # Live variables via per-variable causal-effect rate.
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

    # Probe: learned variable value tables vs GT atoms.
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


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return 0.0, 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return mean, math.sqrt(var)


def _render_markdown(res: dict) -> str:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Seed / basis variance study (Phase A)")
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
    args = p.parse_args()

    task = _load_task(args.task)
    site = _load_site(task, args.site_layer, args.device)

    per_seed: list[dict] = []
    for seed in range(args.seeds):
        causal_model, rotation, layout, config, final = _train_joint(
            task, site, args, seed
        )
        cls = _classify_seed(
            task, site, causal_model, rotation, layout, config, args, seed
        )
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
    md_path.write_text(_render_markdown(res))
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
