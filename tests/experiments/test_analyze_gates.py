"""Tests for the night-3 gate-sweep analyzer (``experiments.analyze_gates``).

Covers partial-directory robustness (missing dir, malformed JSON,
schema-incomplete JSON are skipped) and table/plot generation from synthetic
result dicts that mirror the ``run_phase_a`` / ``run_phase_b`` gate schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyze_gates import (
    GateRow,
    load_dir,
    load_row,
    make_plot,
    render_lm_md,
    render_toy_md,
)


def _toy_result(
    *, task: str, layer: int, lam: float, seed: int, gated_k: int
) -> dict:
    """A synthetic Phase-A gate run mirroring run_phase_a's JSON schema."""
    return {
        "config": {
            "task": task,
            "method": "joint",
            "site_layer": layer,
            "seed": seed,
            "gates": True,
            "lambda_gate": lam,
            "k_max": 4,
        },
        "recovery_score": 0.9,
        "history": [
            {"step": 0, "gated_k": 4},
            {"step": 500, "gated_k": 3},
            {"step": 1000, "gated_k": gated_k},
            {"step": 1499, "gated_k": gated_k},
        ],
        "final": {
            "iia_1": 0.98,
            "iia_2": 0.95,
            "effective_k": float(gated_k),
            "aligned_dims": 12.0,
            "hard_widths": [3.0, 3.0, 3.0, 3.0],
            "gated_k": gated_k,
            "g_det": [1.0, 1.0, 0.0, 0.0],
            "live_indices": list(range(gated_k)),
            "hard_widths_gated": [3.0] * gated_k + [0.0] * (4 - gated_k),
            "iia_1_live": 0.99,
            "iia_2_live": 0.97,
        },
    }


def _lm_result(*, lam: float, seed: int, gated_k: int) -> dict:
    """A synthetic Phase-B (LM) gate run: uses ``layer`` not ``site_layer``."""
    d = _toy_result(task="price_tagging", layer=17, lam=lam, seed=seed, gated_k=gated_k)
    cfg = d["config"]
    cfg["layer"] = cfg.pop("site_layer")
    cfg["model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    return d


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


# -- robustness ---------------------------------------------------------------


def test_load_dir_missing_directory() -> None:
    """A non-existent directory yields no rows (no exception)."""
    assert load_dir(Path("/nonexistent/night3/gates_toy")) == []


def test_load_dir_skips_malformed_and_incomplete(tmp_path: Path, capsys) -> None:
    """Malformed JSON and schema-incomplete files are skipped with warnings."""
    good = _toy_result(task="hierarchical_equality", layer=1, lam=0.1, seed=0, gated_k=2)
    _write(tmp_path / "hier_l1_lg0.1_s0.json", good)
    # Malformed JSON (truncated).
    (tmp_path / "hier_l1_lg0.1_s1.json").write_text("{ not valid json ")
    # Schema-incomplete: has config but no final.
    _write(tmp_path / "hier_l1_lg0.1_s2.json", {"config": {"task": "x", "site_layer": 1}})
    # Empty file.
    (tmp_path / "hier_l1_lg0.3_s0.json").write_text("")

    rows = load_dir(tmp_path)
    assert len(rows) == 1
    assert rows[0].task == "hierarchical_equality"
    err = capsys.readouterr().err
    assert "skipping" in err


def test_load_row_partial_missing_gate_fields(tmp_path: Path) -> None:
    """A run without gate fields still loads, with None gate columns."""
    d = _toy_result(task="boolean_comp", layer=1, lam=0.0, seed=0, gated_k=4)
    for key in ("gated_k", "iia_1_live", "iia_2_live", "hard_widths_gated"):
        d["final"].pop(key)
    p = tmp_path / "bool_l1_lg0_s0.json"
    _write(p, d)
    row = load_row(p)
    assert row is not None
    assert row.gated_k is None
    assert row.iia_1_live is None
    assert row.aligned_dims_gated is None
    # Non-gate fields still present.
    assert row.iia_1 == pytest.approx(0.98)


def test_prune_step_last_change(tmp_path: Path) -> None:
    """prune_step is the last history step at which gated_k changed."""
    d = _toy_result(task="hierarchical_equality", layer=2, lam=0.03, seed=1, gated_k=2)
    p = tmp_path / "hier_l2_lg0.03_s1.json"
    _write(p, d)
    row = load_row(p)
    assert row is not None
    # gated_k changes 4->3 at 500, 3->2 at 1000, stable after => 1000.
    assert row.prune_step == 1000


def test_layer_resolution_phase_b_schema(tmp_path: Path) -> None:
    """Phase-B configs use ``layer`` and are resolved correctly."""
    d = _lm_result(lam=0.05, seed=0, gated_k=3)
    p = tmp_path / "pt_gates_l17_lg0.05_s0.json"
    _write(p, d)
    row = load_row(p)
    assert row is not None
    assert row.layer == 17


# -- table generation ---------------------------------------------------------


def test_render_toy_md_from_two_results(tmp_path: Path) -> None:
    """Two synthetic runs render a per-(task,layer) table + aggregate."""
    r0 = _toy_result(task="hierarchical_equality", layer=1, lam=0.0, seed=0, gated_k=4)
    r1 = _toy_result(task="hierarchical_equality", layer=1, lam=0.3, seed=1, gated_k=2)
    _write(tmp_path / "hier_l1_lg0_s0.json", r0)
    _write(tmp_path / "hier_l1_lg0.3_s1.json", r1)

    rows = load_dir(tmp_path)
    assert len(rows) == 2
    md = render_toy_md(rows)
    assert "hierarchical_equality (layer 1)" in md
    # Column headers present.
    for col in ("gated_k", "iia_1_live", "recovery_score", "aligned_dims_gated", "prune_step"):
        assert col in md
    # Aggregate block present with both lambdas.
    assert "Per-lambda aggregate" in md
    assert "0.000" in md and "0.300" in md


def test_render_toy_md_empty() -> None:
    """No toy rows -> a placeholder, not a crash."""
    md = render_toy_md([])
    assert "No toy results found" in md


def test_render_lm_md_has_night2_anchors() -> None:
    """LM section always quotes the night-2 anchor block."""
    md = render_lm_md([])
    assert "capped joint" in md
    assert "0.855" in md and "0.863" in md
    assert "capped das_true" in md
    assert "capped random_rotation" in md


def test_make_plot_toy_only(tmp_path: Path) -> None:
    """A PNG is written even when only toy data exists."""
    rows = [
        GateRow(
            path="a.json", task="hierarchical_equality", layer=1, lambda_gate=0.0,
            seed=0, gated_k=4, effective_k=4.0, iia_1_live=0.99, iia_2_live=0.97,
            iia_1=0.98, iia_2=0.95, recovery_score=0.9, aligned_dims_gated=12.0,
            prune_step=1000,
        ),
        GateRow(
            path="b.json", task="hierarchical_equality", layer=1, lambda_gate=0.3,
            seed=0, gated_k=2, effective_k=2.0, iia_1_live=0.9, iia_2_live=0.85,
            iia_1=0.9, iia_2=0.85, recovery_score=0.8, aligned_dims_gated=6.0,
            prune_step=1000,
        ),
    ]
    out = tmp_path / "night3_gates.png"
    result = make_plot(rows, [], out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
