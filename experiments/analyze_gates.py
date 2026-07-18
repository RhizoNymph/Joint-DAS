"""Night-3 hard-concrete gate-sweep analyzer.

Aggregates the two night-3 gate sweeps into a markdown summary + a figure:

1. **Toy** — ``experiments/results/night3/gates_toy/{hier|bool}_l{layer}_lg{lambda}_s{seed}.json``
   (tasks hier l1/l2, bool l1; ``lambda_gate in {0, 0.01, 0.03, 0.1, 0.3}``;
   seeds 0-2; joint, k_max=4, 1500 steps).
2. **LM** — ``experiments/results/night3/gates_lm/pt_gates_l17_lg{lambda}_s{seed}.json``
   (``lambda_gate in {0, 0.01, 0.05, 0.2}``; Qwen2.5-1.5B l17 capped recipe).

Files land incrementally (this runs mid-sweep), so loading is defensive: a
missing / malformed / schema-incomplete JSON is skipped with a warning rather
than aborting the run. Everything read from a run JSON is a plain Python float /
int / list (results are produced on CUDA but serialized as JSON) — no tensors.

Per (task, layer) a markdown table is emitted with rows = (lambda_gate, seed)
and columns gated_k, effective_k, iia_1_live, iia_2_live, iia_1, iia_2,
recovery_score, aligned_dims_gated (sum of gate-scaled hard widths), prune_step
(last history step at which gated_k changed). A per-lambda aggregate block
reports mean gated_k and mean live-IIA over seeds. The LM sweep gets one table
plus a night-2 anchor reference block.

Usage
-----
    uv run python experiments/analyze_gates.py \
        --toy-dir experiments/results/night3/gates_toy \
        --lm-dir experiments/results/night3/gates_lm \
        --out-md experiments/results/night3/gates_summary.md \
        --plot docs/assets/night3_gates.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Night-2 capped-LM anchors (layer 17, Qwen2.5-1.5B) quoted in the LM block.
NIGHT2_ANCHORS = [
    ("capped joint", 0.855, 0.863, 4, 59),
    ("capped das_true", 0.891, 0.922, 4, 32),
    ("capped random_rotation", 0.781, 0.730, None, None),
]


@dataclass
class GateRow:
    """One loaded gate-sweep run (a single JSON file)."""

    path: str
    task: str
    layer: int
    lambda_gate: float
    seed: int
    gated_k: float | None
    effective_k: float | None
    iia_1_live: float | None
    iia_2_live: float | None
    iia_1: float | None
    iia_2: float | None
    recovery_score: float | None
    aligned_dims_gated: float | None
    prune_step: int | None


def _f(x) -> float | None:
    """Coerce to float, tolerating None/JSON-null and non-numeric junk."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _resolve_layer(cfg: dict) -> int | None:
    """Layer index across Phase A (``site_layer``) / Phase B (``layer``) schemas."""
    for key in ("site_layer", "layer"):
        if key in cfg and cfg[key] is not None:
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                return None
    return None


def _aligned_dims_gated(fin: dict) -> float | None:
    """Total live-variable subspace width (sum of gate-scaled hard widths)."""
    hwg = fin.get("hard_widths_gated")
    if not isinstance(hwg, list):
        return None
    try:
        return float(sum(float(w) for w in hwg))
    except (TypeError, ValueError):
        return None


def _prune_step(history: list) -> int | None:
    """Last step at which ``gated_k`` changed in the recorded history.

    Returns the step of the final transition (pruning-dynamics settling point),
    or None if history lacks per-step gated_k. When gated_k never changes across
    the recorded steps we report the first recorded step (settled from the
    start).
    """
    if not isinstance(history, list) or not history:
        return None
    pts: list[tuple[int, int]] = []
    for rec in history:
        if not isinstance(rec, dict):
            continue
        gk = rec.get("gated_k")
        st = rec.get("step")
        if gk is None or st is None:
            continue
        try:
            pts.append((int(st), int(gk)))
        except (TypeError, ValueError):
            continue
    if not pts:
        return None
    pts.sort()
    last_change = pts[0][0]
    for i in range(1, len(pts)):
        if pts[i][1] != pts[i - 1][1]:
            last_change = pts[i][0]
    return last_change


def load_row(path: Path) -> GateRow | None:
    """Load one gate-sweep JSON into a :class:`GateRow`, or None if unusable.

    Emits a warning to stderr and returns None on any of: unreadable file,
    malformed JSON, or a schema that lacks ``config``/``final``.
    """
    try:
        text = path.read_text()
    except OSError as exc:  # pragma: no cover - unlikely in tests
        print(f"warning: skipping {path.name}: unreadable ({exc})", file=sys.stderr)
        return None
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        print(f"warning: skipping {path.name}: malformed JSON", file=sys.stderr)
        return None
    if not isinstance(d, dict) or "config" not in d or "final" not in d:
        print(
            f"warning: skipping {path.name}: missing config/final", file=sys.stderr
        )
        return None
    cfg = d.get("config") or {}
    fin = d.get("final") or {}
    if not isinstance(cfg, dict) or not isinstance(fin, dict):
        print(f"warning: skipping {path.name}: bad config/final types", file=sys.stderr)
        return None
    layer = _resolve_layer(cfg)
    if layer is None:
        print(f"warning: skipping {path.name}: no layer in config", file=sys.stderr)
        return None
    return GateRow(
        path=path.name,
        task=str(cfg.get("task", "?")),
        layer=layer,
        lambda_gate=_f(cfg.get("lambda_gate")) or 0.0,
        seed=int(cfg.get("seed", -1)) if cfg.get("seed") is not None else -1,
        gated_k=_f(fin.get("gated_k")),
        effective_k=_f(fin.get("effective_k")),
        iia_1_live=_f(fin.get("iia_1_live")),
        iia_2_live=_f(fin.get("iia_2_live")),
        iia_1=_f(fin.get("iia_1")),
        iia_2=_f(fin.get("iia_2")),
        recovery_score=_f(d.get("recovery_score")),
        aligned_dims_gated=_aligned_dims_gated(fin),
        prune_step=_prune_step(d.get("history", [])),
    )


def load_dir(results_dir: Path) -> list[GateRow]:
    """Load every ``*.json`` in ``results_dir``, skipping unusable files."""
    rows: list[GateRow] = []
    if not results_dir.exists():
        print(f"warning: {results_dir} does not exist; no rows", file=sys.stderr)
        return rows
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        row = load_row(path)
        if row is not None:
            rows.append(row)
    return rows


# -- formatting ---------------------------------------------------------------


def _fmt(x: float | None, prec: int = 3) -> str:
    return "-" if x is None else f"{x:.{prec}f}"


def _fmt_int(x: float | None) -> str:
    return "-" if x is None else f"{int(round(x))}"


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return None if not nums else sum(nums) / len(nums)


_TABLE_COLS = [
    ("lambda_gate", lambda r: _fmt(r.lambda_gate, 3)),
    ("seed", lambda r: str(r.seed)),
    ("gated_k", lambda r: _fmt_int(r.gated_k)),
    ("effective_k", lambda r: _fmt(r.effective_k, 2)),
    ("iia_1_live", lambda r: _fmt(r.iia_1_live)),
    ("iia_2_live", lambda r: _fmt(r.iia_2_live)),
    ("iia_1", lambda r: _fmt(r.iia_1)),
    ("iia_2", lambda r: _fmt(r.iia_2)),
    ("recovery_score", lambda r: _fmt(r.recovery_score)),
    ("aligned_dims_gated", lambda r: _fmt(r.aligned_dims_gated, 2)),
    ("prune_step", lambda r: _fmt_int(r.prune_step)),
]


def _md_table(rows: list[GateRow]) -> list[str]:
    """Per-run rows table (sorted by lambda_gate then seed)."""
    header = [c[0] for c in _TABLE_COLS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in sorted(rows, key=lambda r: (r.lambda_gate, r.seed)):
        lines.append("| " + " | ".join(fmt(r) for _, fmt in _TABLE_COLS) + " |")
    return lines


def _md_aggregate(rows: list[GateRow]) -> list[str]:
    """Per-lambda aggregate: mean gated_k, mean live-IIA over seeds."""
    by_lambda: dict[float, list[GateRow]] = defaultdict(list)
    for r in rows:
        by_lambda[r.lambda_gate].append(r)
    lines = [
        "| lambda_gate | n_seeds | mean gated_k | mean iia_1_live | mean iia_2_live |",
        "|---|---|---|---|---|",
    ]
    for lam in sorted(by_lambda):
        grp = by_lambda[lam]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(lam, 3),
                    str(len(grp)),
                    _fmt(_mean([r.gated_k for r in grp]), 2),
                    _fmt(_mean([r.iia_1_live for r in grp])),
                    _fmt(_mean([r.iia_2_live for r in grp])),
                ]
            )
            + " |"
        )
    return lines


def render_toy_md(rows: list[GateRow]) -> str:
    """Full toy section: one table + aggregate per (task, layer)."""
    lines = ["# Night-3 gate sweeps", "", "## Toy sweeps", ""]
    if not rows:
        lines.append("_No toy results found._")
        return "\n".join(lines) + "\n"
    by_group: dict[tuple[str, int], list[GateRow]] = defaultdict(list)
    for r in rows:
        by_group[(r.task, r.layer)].append(r)
    for (task, layer) in sorted(by_group):
        grp = by_group[(task, layer)]
        lines.append(f"### {task} (layer {layer})  —  {len(grp)} runs")
        lines.append("")
        lines.extend(_md_table(grp))
        lines.append("")
        lines.append("_Per-lambda aggregate (mean over seeds):_")
        lines.append("")
        lines.extend(_md_aggregate(grp))
        lines.append("")
    return "\n".join(lines) + "\n"


def render_lm_md(rows: list[GateRow]) -> str:
    """LM section: night-2 anchor block + one table + aggregate."""
    lines = ["## LM sweep (Qwen2.5-1.5B, layer 17, capped)", ""]
    lines.append("_Night-2 capped anchors (reference):_")
    lines.append("")
    lines.append("| method | iia_1 | iia_2 | k_eff | aligned dims |")
    lines.append("|---|---|---|---|---|")
    for name, i1, i2, keff, dims in NIGHT2_ANCHORS:
        lines.append(
            f"| {name} | {i1:.3f} | {i2:.3f} | "
            f"{'-' if keff is None else keff} | {'-' if dims is None else dims} |"
        )
    lines.append("")
    if not rows:
        lines.append("_No LM results found yet._")
        return "\n".join(lines) + "\n"
    by_group: dict[tuple[str, int], list[GateRow]] = defaultdict(list)
    for r in rows:
        by_group[(r.task, r.layer)].append(r)
    for (task, layer) in sorted(by_group):
        grp = by_group[(task, layer)]
        lines.append(f"### gates {task} (layer {layer})  —  {len(grp)} runs")
        lines.append("")
        lines.extend(_md_table(grp))
        lines.append("")
        lines.append("_Per-lambda aggregate (mean over seeds):_")
        lines.append("")
        lines.extend(_md_aggregate(grp))
        lines.append("")
    return "\n".join(lines) + "\n"


# -- plotting -----------------------------------------------------------------


def _panel_vs_lambda(ax, rows: list[GateRow], value, ylabel: str, title: str) -> None:
    """Seed-averaged line per (task,layer) with per-seed scatter, vs lambda."""
    by_group: dict[tuple[str, int], list[GateRow]] = defaultdict(list)
    for r in rows:
        by_group[(r.task, r.layer)].append(r)
    for (task, layer), grp in sorted(by_group.items()):
        by_lambda: dict[float, list[float]] = defaultdict(list)
        for r in grp:
            v = value(r)
            if v is not None:
                by_lambda[r.lambda_gate].append(v)
        if not by_lambda:
            continue
        lams = sorted(by_lambda)
        means = [sum(by_lambda[l]) / len(by_lambda[l]) for l in lams]
        label = f"{task} l{layer}"
        (line,) = ax.plot(lams, means, marker="o", label=label)
        color = line.get_color()
        for lam in lams:
            for v in by_lambda[lam]:
                ax.scatter([lam], [v], color=color, alpha=0.35, s=18, zorder=3)
    ax.set_xlabel("lambda_gate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7)


def make_plot(toy_rows: list[GateRow], lm_rows: list[GateRow], out: Path) -> Path:
    """Figure: gated_k + live-IIA vs lambda for toy (and LM if present)."""
    have_lm = bool(lm_rows)
    nrows = 2 if have_lm else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.4 * nrows), squeeze=False)

    _panel_vs_lambda(
        axes[0][0], toy_rows, lambda r: r.gated_k,
        "gated_k", "Toy: gated_k vs lambda_gate",
    )
    _panel_vs_lambda(
        axes[0][1], toy_rows, lambda r: r.iia_1_live,
        "iia_1_live", "Toy: live IIA (|I|=1) vs lambda_gate",
    )
    if have_lm:
        _panel_vs_lambda(
            axes[1][0], lm_rows, lambda r: r.gated_k,
            "gated_k", "LM: gated_k vs lambda_gate",
        )
        _panel_vs_lambda(
            axes[1][1], lm_rows, lambda r: r.iia_1_live,
            "iia_1_live", "LM: live IIA (|I|=1) vs lambda_gate",
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- main ---------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Night-3 gate-sweep analyzer")
    p.add_argument("--toy-dir", default="experiments/results/night3/gates_toy")
    p.add_argument("--lm-dir", default="experiments/results/night3/gates_lm")
    p.add_argument("--out-md", default="experiments/results/night3/gates_summary.md")
    p.add_argument("--plot", default="docs/assets/night3_gates.png")
    args = p.parse_args()

    toy_rows = load_dir(Path(args.toy_dir))
    lm_rows = load_dir(Path(args.lm_dir))

    md = render_toy_md(toy_rows) + "\n" + render_lm_md(lm_rows)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    plot_path = make_plot(toy_rows, lm_rows, Path(args.plot))

    print(f"loaded {len(toy_rows)} toy + {len(lm_rows)} LM runs")
    print(f"wrote {out_md}")
    print(f"wrote {plot_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
