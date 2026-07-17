"""Night-2 analysis plots for Joint-DAS.

Emits four PNGs (dpi 150, matplotlib Agg, no seaborn) into an assets dir:

- ``night2_capped_lm.png``     capped-LM bar comparison (joint vs control
                               [vs das_true when present]), iia_1 + iia_2, k_eff
                               annotated.
- ``night2_seed_basis.png``    which boolean functions the live variables realise
                               across the 10 seeds (basis-composition histogram).
- ``night2_search_hier.png``   hier-eq search ranking (horizontal bars, E1+E2
                               highlighted last).
- ``night2_wrong_and.png``     wrong-composition measured iia vs analytic ceiling.

Reads the fixed night-2 result JSONs; run from repo root:

    uv run python experiments/analyze_night2.py \
        --night2-dir experiments/results/night2 \
        --assets-dir docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# -- 1. capped-LM comparison ---------------------------------------------------


def plot_capped_lm(night2: Path, assets: Path) -> Path:
    """Grouped iia_1/iia_2 bars for capped-LM methods, k_eff annotated."""
    specs = [
        ("joint\n(capped)", "pt_joint_l17_capped.json", "#1f77b4"),
        ("random\n(control)", "pt_random_l17_capped.json", "#9467bd"),
        ("joint z-digits\n(capped)", "pt_joint_l17_zdigits_capped.json", "#17becf"),
        ("das_true\n(capped)", "pt_das_true_l17_capped.json", "#2ca02c"),
    ]
    rows = []
    for label, fname, color in specs:
        p = night2 / fname
        if not p.exists():
            continue
        d = _load(p)
        fin = d["final"]
        rows.append((label, color, fin["iia_1"], fin["iia_2"], fin["effective_k"]))

    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(rows)), 4.2))
    bar_w = 0.38
    xs = list(range(len(rows)))
    iia1 = [r[2] for r in rows]
    iia2 = [r[3] for r in rows]
    ax.bar(
        [x - bar_w / 2 for x in xs], iia1, bar_w, label="iia_1",
        color=[r[1] for r in rows], alpha=0.9, edgecolor="black", linewidth=0.4,
    )
    ax.bar(
        [x + bar_w / 2 for x in xs], iia2, bar_w, label="iia_2",
        color=[r[1] for r in rows], alpha=0.5, edgecolor="black", linewidth=0.4,
        hatch="//",
    )
    for x, r in zip(xs, rows):
        ax.text(x, max(r[2], r[3]) + 0.02, f"k_eff={r[4]}", ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("IIA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Capped-LM (Qwen2.5-1.5B, layer 17): iia_1 (solid) / iia_2 (hatched)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = assets / "night2_capped_lm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- 2. seed-study basis composition ------------------------------------------


def plot_seed_basis(night2: Path, assets: Path) -> Path:
    """Histogram of which boolean fns the live variables realise across seeds."""
    d = _load(night2 / "seed_study_hier_l1.json")
    fn_order = [
        "A", "B", "notA", "notB", "AND", "OR", "NAND", "NOR", "XOR", "XNOR",
    ]
    counts = {f: 0 for f in fn_order}
    for s in d["per_seed"]:
        for lf in s["live_fns"]:
            counts[lf["fn"]] = counts.get(lf["fn"], 0) + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(len(fn_order)))
    ys = [counts[f] for f in fn_order]
    # colour the two GT atoms (A,B) distinctly from composite gates.
    colors = ["#2ca02c" if f in ("A", "B") else "#1f77b4" for f in fn_order]
    ax.bar(xs, ys, color=colors, alpha=0.9, edgecolor="black", linewidth=0.4)
    for x, y in zip(xs, ys):
        if y:
            ax.text(x, y + 0.05, str(y), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(fn_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# live variables across 10 seeds")
    ax.set_title(
        "Seed study (hier-eq, layer 1): boolean fns realised by live variables\n"
        "green = GT atoms (A,B); blue = composite gates"
    )
    ax.margins(y=0.15)
    fig.tight_layout()
    out = assets / "night2_seed_basis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- 3. hier search ranking ----------------------------------------------------


def plot_search_hier(night2: Path, assets: Path) -> Path:
    """Horizontal bar chart of hier-eq search iia_1, E1+E2 highlighted last."""
    d = _load(night2 / "search_hier_l1.json")
    ranking = d["ranking"]
    labels = [f"{r['V1']}+{r['V2']}" for r in ranking]
    iia1 = [r["iia_1"] for r in ranking]
    is_e1e2 = [
        {r["V1"], r["V2"]} == {"E1", "E2"} for r in ranking
    ]
    # plot best at top: reverse so rank 1 sits at top.
    order = list(range(len(ranking)))[::-1]
    ys = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for y, idx in zip(ys, order):
        color = "#d62728" if is_e1e2[idx] else "#1f77b4"
        ax.barh(y, iia1[idx], color=color, alpha=0.9, edgecolor="black", linewidth=0.4)
        ax.text(iia1[idx] + 0.01, y, f"{iia1[idx]:.3f}", va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("iia_1 (fit Q per candidate pair)")
    ax.set_xlim(0, 1.1)
    ax.set_title(
        "Search baseline (hier-eq, layer 1): E1+E2 (red) ranks LAST;\n"
        "composite bases achieve perfect IIA"
    )
    fig.tight_layout()
    out = assets / "night2_search_hier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- 4. wrong-and measured vs ceiling -----------------------------------------


def _group_mean(night2: Path, prefix: str) -> tuple[float, float, float]:
    """Mean iia_1, iia_2 and analytic |I|=1 ceiling over the 3 seeds of a group."""
    iia1, iia2, ceil = [], [], []
    for s in range(3):
        d = _load(night2 / f"{prefix}_s{s}.json")
        iia1.append(d["final"]["iia_1"])
        iia2.append(d["final"]["iia_2"])
        ceil.append(d["agreement_ceiling"]["1"])
    n = len(iia1)
    return (sum(iia1) / n, sum(iia2) / n, sum(ceil) / n)


def plot_wrong_and(night2: Path, assets: Path) -> Path:
    """Measured iia_1/iia_2 vs analytic agreement ceiling for wrong-AND models."""
    groups = [
        ("hier L0", "hier_das_wrong_and_l0"),
        ("hier L1", "hier_das_wrong_and_l1"),
        ("bool L0", "bool_das_wrong_and_l0"),
        ("bool L1", "bool_das_wrong_and_l1"),
    ]
    data = [(lbl, *_group_mean(night2, pre)) for lbl, pre in groups]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = list(range(len(data)))
    bar_w = 0.35
    ax.bar(
        [x - bar_w / 2 for x in xs], [d[1] for d in data], bar_w,
        label="iia_1 (measured)", color="#d62728", alpha=0.9,
        edgecolor="black", linewidth=0.4,
    )
    ax.bar(
        [x + bar_w / 2 for x in xs], [d[2] for d in data], bar_w,
        label="iia_2 (measured)", color="#d62728", alpha=0.5,
        edgecolor="black", linewidth=0.4, hatch="//",
    )
    # ceiling as a horizontal marker per group.
    for x, d in zip(xs, data):
        ax.plot(
            [x - 0.45, x + 0.45], [d[3], d[3]], color="black", lw=1.6,
            label="analytic ceiling" if x == 0 else None,
        )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylabel("IIA")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Wrong-composition (das_wrong_and): measured IIA at-or-below analytic ceiling"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = assets / "night2_wrong_and.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Night-2 Joint-DAS plots")
    p.add_argument("--night2-dir", default="experiments/results/night2")
    p.add_argument("--assets-dir", default="docs/assets")
    args = p.parse_args()
    night2 = Path(args.night2_dir)
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    outs = [
        plot_capped_lm(night2, assets),
        plot_seed_basis(night2, assets),
        plot_search_hier(night2, assets),
        plot_wrong_and(night2, assets),
    ]
    print("wrote:")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
