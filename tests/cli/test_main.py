"""argparse-tree dispatch smoke tests + sweep-status counting.

Every subcommand is exercised at the ``--help`` level (which exits 0) so we
confirm the tree is wired without ever training a model or contacting a host.
The dispatch namespaces are also inspected to confirm each ``run``/``analyze``
leaf records the correct handler.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from jdas.cli import build_parser, main
from jdas.cli.config import EnvConfig
from jdas.cli import sweeps as sweeps_mod

_SPEC_DIR = Path(__file__).resolve().parents[2] / "experiments" / "sweeps"

_HELP_INVOCATIONS = [
    ["--help"],
    ["run", "--help"],
    ["run", "phase-a", "--help"],
    ["run", "phase-b", "--help"],
    ["run", "search", "--help"],
    ["run", "seed-study", "--help"],
    ["analyze", "--help"],
    ["analyze", "gates", "--help"],
    ["analyze", "phase-a", "--help"],
    ["analyze", "night2", "--help"],
    ["sweep", "--help"],
    ["sweep", "run", "--help"],
    ["sweep", "status", "--help"],
    ["sweep", "collect", "--help"],
    ["cluster", "--help"],
    ["cluster", "sync", "--help"],
    ["cluster", "status", "--help"],
    ["cluster", "exec", "--help"],
    ["cluster", "kill", "--help"],
]


@pytest.mark.parametrize("argv", _HELP_INVOCATIONS, ids=lambda a: " ".join(a))
def test_help_exits_zero(argv):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 0


# -- leaf dispatch wiring -----------------------------------------------------


def test_run_phase_a_leaf_records_handler():
    from jdas.cli import runners

    parser = build_parser()
    args = parser.parse_args(
        ["run", "phase-a", "--task", "hierarchical_equality", "--method", "joint"]
    )
    assert args._run_fn is runners.run_phase_a
    assert args.task == "hierarchical_equality"
    assert args.method == "joint"


def test_run_phase_b_leaf_records_handler():
    from jdas.cli import runners

    parser = build_parser()
    args = parser.parse_args(["run", "phase-b", "--layer", "17", "--method", "das_true"])
    assert args._run_fn is runners.run_phase_b
    assert args.layer == 17


def test_analyze_gates_leaf_records_handler():
    from jdas.cli import analyze

    parser = build_parser()
    args = parser.parse_args(["analyze", "gates"])
    assert args._analyze_fn is analyze.run_gates


def test_cluster_kill_parses_pattern():
    parser = build_parser()
    args = parser.parse_args(["cluster", "kill", "run_phase_b"])
    assert args.cluster_cmd == "kill"
    assert args.pattern == "run_phase_b"


def test_cluster_exec_remainder():
    parser = build_parser()
    args = parser.parse_args(["cluster", "exec", "--", "nvidia-smi", "-L"])
    assert args.cluster_cmd == "exec"
    assert args.cmd == ["--", "nvidia-smi", "-L"]


def test_global_config_flag():
    parser = build_parser()
    args = parser.parse_args(["--config", "x.toml", "cluster", "status"])
    assert args.config == "x.toml"


# -- sweep status counting ----------------------------------------------------


def _cfg_at(tmp_path) -> EnvConfig:
    cfg = EnvConfig()
    return replace(
        cfg,
        paths=replace(cfg.paths, results=str(tmp_path / "results"), logs=str(tmp_path / "logs")),
    )


def test_sweep_status_counts_present(tmp_path, monkeypatch, capsys):
    cfg = _cfg_at(tmp_path)
    monkeypatch.setattr("jdas.cli.load_config", lambda *a, **k: cfg)
    spec = sweeps_mod.load_spec(_SPEC_DIR / "gates_lm_v3.toml")
    runs = sweeps_mod.expand(spec)
    # Create 3 of 7 outputs.
    for run in runs[:3]:
        p = run.out_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    # A failures log line.
    logs = Path(cfg.paths.logs)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{spec.name}_failures.log").write_text("FAILED pt_gates3_l17_lg0.2_s1\n")
    rc = main(["sweep", "status", str(_SPEC_DIR / "gates_lm_v3.toml")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3/7 outputs present" in out
    assert "FAILED pt_gates3_l17_lg0.2_s1" in out


def test_sweep_dry_run_cluster_assignment(tmp_path, monkeypatch, capsys):
    cfg = _cfg_at(tmp_path)
    monkeypatch.setattr("jdas.cli.load_config", lambda *a, **k: cfg)
    rc = main(
        ["sweep", "run", str(_SPEC_DIR / "gates_toy_v3.toml"), "--where", "cluster", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "45 runs" in out
    assert "host assignment" in out
    # 45 over 3 hosts -> 15 each.
    assert "node0: 15 runs" in out
    assert "node1: 15 runs" in out
    assert "node2: 15 runs" in out
    # A rendered filename appears.
    assert "hier_l1_lg0_s0.json" in out


def test_sweep_run_cluster_does_not_ssh(tmp_path, monkeypatch):
    """Dry-run path must never invoke the effectful cluster helper."""
    cfg = _cfg_at(tmp_path)
    monkeypatch.setattr("jdas.cli.load_config", lambda *a, **k: cfg)

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("cluster subprocess invoked during dry-run")

    monkeypatch.setattr("jdas.cli.cluster._run", _boom)
    rc = main(
        ["sweep", "run", str(_SPEC_DIR / "gates_lm_v3.toml"), "--where", "cluster", "--dry-run"]
    )
    assert rc == 0
