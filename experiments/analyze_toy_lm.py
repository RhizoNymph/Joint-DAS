"""Aggregate toy-model / LM result JSONs into a summary table and plots.

Reads a directory of result JSONs (as written by ``jdas run toy`` / ``jdas run
lm``), groups by ``(task, method, layer)``, and emits:

- (a) a markdown summary table of mean +/- std over seeds for iia_1, iia_2,
  effective_k, recovery_score, refit_iia_1, refit_iia_2 (written to a ``.md``
  file and printed);
- (b) plots into ``docs/assets/``:
    * per task: grouped bar chart of iia_1 and iia_2 by (layer x method) with
      per-seed scatter overlaid;
    * per task: recovery-score bar chart by (layer x method);
    * per task: training-curve (IIA vs step) for one representative joint run.

Schema tolerance
----------------
Toy-model configs use ``site_layer``; LM configs use ``layer`` (plus
``model``/``template_id``). Both are handled: the "layer" key is resolved from
whichever of ``site_layer``/``layer`` is present. The result schema is otherwise
identical (``config``, ``final.{iia_1,iia_2,effective_k,hard_widths}``, optional
``recovery_score``/``refit_iia_1``/``refit_iia_2``, ``history``).

Usage (via the CLI: ``jdas analyze toy --results-dir ... --out-md ...``)
-----
    uv run python experiments/analyze_toy_lm.py \
        --results-dir experiments/results/phase_a \
        --out-md experiments/results/phase_a_summary.md \
        --assets-dir docs/assets --tag phase_a
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _plt():
    """Lazily import matplotlib with a headless backend.

    Kept lazy so the aggregation / markdown-table path can run in environments
    without a usable matplotlib (the plotting path imports it on demand).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# -- loading -------------------------------------------------------------------


def _resolve_layer(cfg: dict) -> int:
    """Resolve the intervention-site layer index across toy/LM schemas."""
    if "site_layer" in cfg:
        return int(cfg["site_layer"])
    if "layer" in cfg:
        return int(cfg["layer"])
    raise KeyError("result config has neither 'site_layer' nor 'layer'")


def load_results(results_dir: Path) -> list[dict]:
    """Load and normalize every ``*.json`` result under ``results_dir``."""
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json") or path.name.startswith("introspect_"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "config" not in d or "final" not in d:
            continue
        cfg = d["config"]
        fin = d["final"]
        rows.append(
            {
                "path": path.name,
                "task": cfg.get("task", "?"),
                "method": cfg.get("method", "?"),
                "layer": _resolve_layer(cfg),
                "seed": int(cfg.get("seed", -1)),
                "iia_1": _f(fin.get("iia_1")),
                "iia_2": _f(fin.get("iia_2")),
                "effective_k": _f(fin.get("effective_k")),
                "recovery_score": _f(d.get("recovery_score")),
                "refit_iia_1": _f(d.get("refit_iia_1")),
                "refit_iia_2": _f(d.get("refit_iia_2")),
                "hard_widths": fin.get("hard_widths"),
                "history": d.get("history", []),
            }
        )
    return rows


def _f(x) -> float | None:
    return None if x is None else float(x)


# -- aggregation ---------------------------------------------------------------

METRICS = ["iia_1", "iia_2", "effective_k", "recovery_score", "refit_iia_1", "refit_iia_2"]


def _mean_std(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan, 0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0, 1)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return (mean, math.sqrt(var), n)


def aggregate(rows: list[dict]) -> dict[tuple, dict]:
    """Group rows by ``(task, method, layer)`` and reduce each metric to
    ``(mean, std, n)``.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["method"], r["layer"])].append(r)
    agg: dict[tuple, dict] = {}
    for key, grp in groups.items():
        stats = {m: _mean_std([g[m] for g in grp]) for m in METRICS}
        agg[key] = {"stats": stats, "n_seeds": len(grp)}
    return agg


# -- markdown table ------------------------------------------------------------


def _fmt(ms: tuple[float, float, int]) -> str:
    mean, std, n = ms
    if n == 0 or math.isnan(mean):
        return "-"
    if n == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def render_markdown(agg: dict[tuple, dict], tag: str) -> str:
    lines = [f"# {tag} summary\n"]
    lines.append("Mean±std over seeds (n in last column). `-` = metric not applicable.\n")
    header = (
        "| task | method | layer | iia_1 | iia_2 | eff_k | recovery | "
        "refit_iia_1 | refit_iia_2 | n |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(agg.keys()):
        task, method, layer = key
        s = agg[key]["stats"]
        lines.append(
            f"| {task} | {method} | {layer} | "
            f"{_fmt(s['iia_1'])} | {_fmt(s['iia_2'])} | {_fmt(s['effective_k'])} | "
            f"{_fmt(s['recovery_score'])} | {_fmt(s['refit_iia_1'])} | "
            f"{_fmt(s['refit_iia_2'])} | {agg[key]['n_seeds']} |"
        )
    return "\n".join(lines) + "\n"


# -- plotting ------------------------------------------------------------------

_METHOD_ORDER = ["joint", "das_true", "das_wrong", "random_rotation"]
_METHOD_COLORS = {
    "joint": "#1f77b4",
    "das_true": "#2ca02c",
    "das_wrong": "#d62728",
    "random_rotation": "#9467bd",
}


def _methods_present(rows: list[dict]) -> list[str]:
    present = {r["method"] for r in rows}
    ordered = [m for m in _METHOD_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _layers_present(rows: list[dict]) -> list[int]:
    return sorted({r["layer"] for r in rows})


def plot_iia_bars(
    rows: list[dict], task: str, metric: str, assets: Path, tag: str
) -> Path:
    """Grouped bar chart of ``metric`` by (layer x method) with seed scatter."""
    plt = _plt()
    task_rows = [r for r in rows if r["task"] == task]
    methods = _methods_present(task_rows)
    layers = _layers_present(task_rows)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(layers) * len(methods) / 2), 4))
    group_w = 0.8
    bar_w = group_w / max(len(methods), 1)
    for mi, method in enumerate(methods):
        means, xs, seed_x, seed_y = [], [], [], []
        for li, layer in enumerate(layers):
            grp = [
                r[metric]
                for r in task_rows
                if r["method"] == method and r["layer"] == layer and r[metric] is not None
            ]
            x = li + (mi - (len(methods) - 1) / 2) * bar_w
            xs.append(x)
            means.append(sum(grp) / len(grp) if grp else 0.0)
            for v in grp:
                seed_x.append(x)
                seed_y.append(v)
        ax.bar(
            xs, means, bar_w * 0.9, label=method,
            color=_METHOD_COLORS.get(method, "#7f7f7f"), alpha=0.85, edgecolor="black",
            linewidth=0.4,
        )
        ax.scatter(seed_x, seed_y, s=14, color="black", zorder=3, alpha=0.7)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task}: {metric} by layer x method")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    out = assets / f"{tag}_{task}_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_recovery(rows: list[dict], task: str, assets: Path, tag: str) -> Path | None:
    """Recovery-score bar chart by (layer x method) with seed scatter."""
    task_rows = [
        r for r in rows if r["task"] == task and r["recovery_score"] is not None
    ]
    if not task_rows:
        return None
    plt = _plt()
    methods = _methods_present(task_rows)
    layers = _layers_present(task_rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(layers) * len(methods) / 2), 4))
    bar_w = 0.8 / max(len(methods), 1)
    for mi, method in enumerate(methods):
        means, xs, sx, sy = [], [], [], []
        for li, layer in enumerate(layers):
            grp = [
                r["recovery_score"]
                for r in task_rows
                if r["method"] == method and r["layer"] == layer
            ]
            x = li + (mi - (len(methods) - 1) / 2) * bar_w
            xs.append(x)
            means.append(sum(grp) / len(grp) if grp else 0.0)
            for v in grp:
                sx.append(x)
                sy.append(v)
        ax.bar(
            xs, means, bar_w * 0.9, label=method,
            color=_METHOD_COLORS.get(method, "#7f7f7f"), alpha=0.85,
            edgecolor="black", linewidth=0.4,
        )
        ax.scatter(sx, sy, s=14, color="black", zorder=3, alpha=0.7)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("recovery_score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task}: GT recovery score by layer x method")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    out = assets / f"{tag}_{task}_recovery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_training_curve(
    rows: list[dict], task: str, assets: Path, tag: str
) -> Path | None:
    """IIA-vs-step curve for one representative joint run (best final iia_1)."""
    cand = [
        r for r in rows
        if r["task"] == task and r["method"] == "joint" and r["history"]
    ]
    if not cand:
        return None
    plt = _plt()
    rep = max(cand, key=lambda r: (r["iia_1"] or 0.0))
    hist = rep["history"]
    steps = [h["step"] for h in hist]
    iia1 = [h.get("iia_1") for h in hist]
    iia2 = [h.get("iia_2") for h in hist]
    effk = [h.get("effective_k") for h in hist]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, iia1, "-o", color="#1f77b4", label="iia_1", ms=4)
    ax.plot(steps, iia2, "-s", color="#ff7f0e", label="iia_2", ms=4)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("training step")
    ax.set_ylabel("IIA")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task}: joint training curve (seed {rep['seed']}, layer {rep['layer']})")

    ax2 = ax.twinx()
    ax2.plot(steps, effk, "-^", color="#2ca02c", label="effective_k", ms=4, alpha=0.6)
    ax2.set_ylabel("effective_k", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = assets / f"{tag}_{task}_train_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- main ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate toy-model / LM results")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--assets-dir", default="docs/assets")
    p.add_argument("--tag", default="phase_a")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_dir)
    if not rows:
        raise SystemExit(f"no result JSONs found under {results_dir}")

    agg = aggregate(rows)
    md = render_markdown(agg, args.tag)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(md)

    tasks = sorted({r["task"] for r in rows})
    plots: list[str] = []
    for task in tasks:
        for metric in ("iia_1", "iia_2"):
            plots.append(str(plot_iia_bars(rows, task, metric, assets, args.tag)))
        rp = plot_recovery(rows, task, assets, args.tag)
        if rp:
            plots.append(str(rp))
        tc = plot_training_curve(rows, task, assets, args.tag)
        if tc:
            plots.append(str(tc))

    print(f"\nwrote {out_md}")
    print("plots:")
    for pth in plots:
        print(f"  {pth}")


if __name__ == "__main__":
    main()
