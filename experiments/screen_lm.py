"""Screen an HF instruct model on the price-tagging task (zero-shot).

For each prompt-template variant, measures how well the frozen model already
solves the yes/no task *without any training* -- this decides which
(model, template, layer) to use for Phase B DAS runs.  We need a template where
the model's yes/no head is already accurate so the intervention has real
behavior to align to.

Reported per template:
- overall accuracy of ``argmax([logit_no, logit_yes])`` vs. the true label;
- per-region accuracy (Z below X / inside [X, Y] / above Y), so we can see the
  model isn't just always answering "no".

Template variants screened (indices into
:data:`jdas.tasks.price_tagging._TEMPLATES`), each with plain and chat rendering:
they are described inline below.  Results are written to
``experiments/results/screen_<model>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jdas.models.hf import load_hf_site
from jdas.tasks.price_tagging import _TEMPLATES, PriceTaggingTask


def _region(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Region label ``0=below X``, ``1=inside``, ``2=above Y``  ``(N,)``."""
    below = z < x
    above = z > y
    region = torch.ones_like(z, dtype=torch.long)  # inside
    region[below] = 0
    region[above] = 2
    return region


@torch.no_grad()
def screen_template(
    site,
    tokenizer,
    template_id: int,
    use_chat: bool,
    n: int,
    device: str,
    seed: int = 0,
    batch: int = 32,
) -> dict[str, object]:
    """Zero-shot accuracy of one (template, chat/plain) variant over ``n`` items."""
    task = PriceTaggingTask(
        tokenizer, template_id=template_id, device=device, use_chat_template=use_chat
    )
    gen = torch.Generator(device=device).manual_seed(seed)

    total_correct = 0
    total = 0
    # per-region correct / count
    reg_correct = [0, 0, 0]
    reg_total = [0, 0, 0]
    n_yes_pred = 0

    done = 0
    while done < n:
        b = min(batch, n - done)
        x, y, z = task._sample_xyz(b, gen, torch.device(device))
        inputs = task._render_batch(x, y, z, torch.device(device))
        labels = task._labels_from_xyz(x, y, z)
        preds = site.logits(inputs).argmax(-1).cpu()
        labels = labels.cpu()
        correct = preds == labels
        total_correct += int(correct.sum())
        total += b
        n_yes_pred += int((preds == 1).sum())
        regions = _region(x, y, z).cpu()
        for r in range(3):
            sel = regions == r
            reg_correct[r] += int(correct[sel].sum())
            reg_total[r] += int(sel.sum())
        done += b

    per_region = {
        name: (reg_correct[r] / reg_total[r] if reg_total[r] else 0.0)
        for r, name in enumerate(("below", "inside", "above"))
    }
    return {
        "template_id": template_id,
        "chat": use_chat,
        "accuracy": total_correct / max(total, 1),
        "per_region": per_region,
        "frac_yes_pred": n_yes_pred / max(total, 1),
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen a model on price tagging")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--templates", type=str, default="all")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--layer", type=int, default=0, help="unused for logits, kept for site build")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Layer only matters for interventions; screening uses clean logits, so
    # any valid layer works to build the site.
    site = load_hf_site(
        args.model, args.layer, args.device, local_files_only=args.local_files_only
    )
    tokenizer = site.tokenizer

    if args.templates == "all":
        template_ids = list(range(len(_TEMPLATES)))
    else:
        template_ids = [int(t) for t in args.templates.split(",")]

    results: list[dict[str, object]] = []
    for tid in template_ids:
        for use_chat in (False, True):
            res = screen_template(
                site, tokenizer, tid, use_chat, args.n, args.device, seed=args.seed
            )
            results.append(res)
            reg = res["per_region"]
            print(
                f"template {tid} chat={int(use_chat)}: "
                f"acc={res['accuracy']:.3f} "
                f"below={reg['below']:.3f} inside={reg['inside']:.3f} "
                f"above={reg['above']:.3f} frac_yes={res['frac_yes_pred']:.3f}"
            )

    best = max(results, key=lambda r: r["accuracy"])
    payload = {"model": args.model, "n": args.n, "results": results, "best": best}

    model_slug = args.model.replace("/", "_")
    out = args.out or f"experiments/results/screen_{model_slug}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
