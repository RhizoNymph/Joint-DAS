"""Tests for sweep spec parsing, grid expansion, rendering, host assignment,
skip-existing/resume, and generated driver-script content."""

from __future__ import annotations

from pathlib import Path

import pytest

from jdas.cli.config import EnvConfig
from jdas.cli.sweeps import (
    SweepError,
    assign_hosts,
    expand,
    load_spec,
    pending_runs,
    render_driver,
    render_out,
    run_argv,
    spec_from_dict,
)

_SPEC_DIR = Path(__file__).resolve().parents[2] / "experiments" / "sweeps"


def _toy_spec_dict() -> dict:
    return {
        "sweep": {
            "name": "t",
            "runner": "phase-a",
            "out_dir": "night3/gates_toy_v3",
            "out_pattern": "{task_short}_l{site_layer}_lg{lambda_gate}_s{seed}.json",
        },
        "grid": {
            "task+site_layer": [
                ["hierarchical_equality", 1],
                ["hierarchical_equality", 2],
                ["boolean_comp", 1],
            ],
            "lambda_gate": [0, 0.01, 0.03, 0.1, 0.3],
            "seed": [0, 1, 2],
        },
        "fixed": {"method": "joint", "gates": True, "steps": 1500},
    }


# -- parsing ------------------------------------------------------------------


def test_spec_missing_sweep_table():
    with pytest.raises(SweepError):
        spec_from_dict({"grid": {}})


def test_spec_bad_runner():
    d = _toy_spec_dict()
    d["sweep"]["runner"] = "nope"
    with pytest.raises(SweepError):
        spec_from_dict(d)


def test_spec_zip_axis_wrong_arity():
    d = _toy_spec_dict()
    d["grid"]["task+site_layer"] = [["hierarchical_equality"]]  # 1-tuple, needs 2
    with pytest.raises(SweepError):
        spec_from_dict(d)


# -- expansion ----------------------------------------------------------------


def test_expand_count_and_zip():
    spec = spec_from_dict(_toy_spec_dict())
    runs = expand(spec)
    assert len(runs) == 3 * 5 * 3  # 45
    # Zipped axis keeps task+layer paired.
    first = runs[0].args
    assert first["task"] == "hierarchical_equality" and first["site_layer"] == 1
    assert first["method"] == "joint" and first["gates"] is True


def test_expand_deterministic():
    spec = spec_from_dict(_toy_spec_dict())
    a = [(r.index, r.out_rel) for r in expand(spec)]
    b = [(r.index, r.out_rel) for r in expand(spec)]
    assert a == b


def test_out_pattern_task_short():
    assert render_out(
        "{task_short}_l{site_layer}_lg{lambda_gate}_s{seed}.json",
        {"task": "hierarchical_equality", "site_layer": 1, "lambda_gate": 0.03, "seed": 2},
    ) == "hier_l1_lg0.03_s2.json"
    assert render_out(
        "{task_short}.json", {"task": "boolean_comp"}
    ) == "bool.json"


def test_out_pattern_missing_field_errors():
    with pytest.raises(SweepError):
        render_out("{nope}.json", {"task": "boolean_comp"})


def test_expand_out_rel_under_out_dir():
    spec = spec_from_dict(_toy_spec_dict())
    runs = expand(spec)
    assert runs[0].out_rel == "night3/gates_toy_v3/hier_l1_lg0_s0.json"


# -- host assignment ----------------------------------------------------------


def test_host_assignment_round_robin_deterministic():
    spec = spec_from_dict(_toy_spec_dict())
    runs = expand(spec)
    hosts = ("node0", "node1", "node2")
    buckets = assign_hosts(runs, hosts)
    # 45 runs over 3 hosts -> 15 each.
    assert [len(buckets[h]) for h in hosts] == [15, 15, 15]
    # host = index % len(hosts).
    for run in runs:
        expected = hosts[run.index % 3]
        assert run in buckets[expected]
    # Deterministic.
    assert assign_hosts(expand(spec), hosts) == buckets


# -- run_argv -----------------------------------------------------------------


def test_run_argv_flags_and_bools():
    spec = spec_from_dict(_toy_spec_dict())
    run = expand(spec)[0]
    argv = run_argv(spec, run, "out.json")
    assert argv[:2] == ["run", "phase-a"]
    assert "--gates" in argv  # bare bool flag
    # value flag rendered as --flag value
    i = argv.index("--site-layer")
    assert argv[i + 1] == "1"
    assert "--task" in argv and argv[argv.index("--task") + 1] == "hierarchical_equality"
    assert argv[-2:] == ["--out", "out.json"]


def test_run_argv_drops_false_and_none():
    spec = spec_from_dict(
        {
            "sweep": {"name": "t", "runner": "phase-b", "out_dir": "d", "out_pattern": "x.json"},
            "grid": {},
            "fixed": {"gates": False, "gate_lr": None, "layer": 17},
        }
    )
    run = expand(spec)[0]
    argv = run_argv(spec, run, "o.json")
    assert "--gates" not in argv and "--gate-lr" not in argv
    assert "--layer" in argv


# -- skip-existing / resume ---------------------------------------------------


def test_pending_runs_skips_existing(tmp_path):
    spec = spec_from_dict(_toy_spec_dict())
    cfg = EnvConfig()
    # Point results at tmp and create one output.
    from dataclasses import replace

    cfg = replace(cfg, paths=replace(cfg.paths, results=str(tmp_path)))
    runs = expand(spec)
    done = runs[0]
    p = done.out_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    pending = pending_runs(runs, cfg)
    assert done not in pending
    assert len(pending) == len(runs) - 1


# -- driver generation --------------------------------------------------------


def test_render_driver_content():
    spec = spec_from_dict(_toy_spec_dict())
    runs = expand(spec)[:2]
    cfg = EnvConfig()
    driver = render_driver(spec, runs, cfg, "node0")
    # cd into remote_dir and export cluster env (sorted keys).
    assert "cd $HOME/Code/ai/learning-causal-representations" in driver
    assert "export HF_HOME=$HOME/hf-cache HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1" in driver
    # uv path from config.
    assert "~/.local/bin/uv run jdas run phase-a" in driver
    # skip-existing guard + failure + status lines.
    assert "[ -f " in driver
    assert "gates_toy_v3_failures.log" not in driver  # uses spec.name 't'
    assert "t_failures.log" in driver
    assert "echo 'DONE_node0' >>" in driver and "t_status.log" in driver
    # per-run log under paths.logs.
    assert "experiments/logs/t_hier_l1_lg0_s0.log" in driver


def test_render_driver_uses_config_paths():
    from dataclasses import replace

    spec = spec_from_dict(_toy_spec_dict())
    cfg = EnvConfig()
    cfg = replace(
        cfg,
        cluster=replace(cfg.cluster, remote_dir="my/remote", uv_path="/opt/uv"),
        paths=replace(cfg.paths, logs="mylogs"),
    )
    driver = render_driver(spec, expand(spec)[:1], cfg, "node1")
    assert "cd $HOME/my/remote" in driver
    assert "/opt/uv run jdas" in driver
    assert "mylogs/t_status.log" in driver


# -- committed specs load + expand to the documented counts -------------------


def test_committed_gates_toy_v3_expands_to_45():
    spec = load_spec(_SPEC_DIR / "gates_toy_v3.toml")
    runs = expand(spec)
    assert len(runs) == 45
    names = {Path(r.out_rel).name for r in runs}
    assert "hier_l1_lg0.03_s2.json" in names
    assert "bool_l1_lg0.3_s1.json" in names
    # schedule flags present in every run.
    assert runs[0].args["gate_lr"] == 0.05
    assert runs[0].args["gate_warmup"] == 300
    assert runs[0].args["gate_clamp"] == 3.0


def test_committed_gates_lm_v3_seven_runs():
    spec = load_spec(_SPEC_DIR / "gates_lm_v3.toml")
    runs = expand(spec)
    assert len(runs) == 7
    names = {Path(r.out_rel).name for r in runs}
    assert "pt_gates3_l17_lg0_s0.json" in names
    assert "pt_gates3_l17_lg0.2_s1.json" in names
    assert runs[0].args["gate_warmup"] == 200


def test_committed_gates_toy_v1_no_gate_lr():
    spec = load_spec(_SPEC_DIR / "gates_toy_v1.toml")
    runs = expand(spec)
    assert len(runs) == 45
    assert "gate_lr" not in runs[0].args


def test_committed_gates_v2_has_gate_lr_no_schedule():
    spec = load_spec(_SPEC_DIR / "gates_toy_v2.toml")
    runs = expand(spec)
    assert runs[0].args["gate_lr"] == 0.05
    assert "gate_warmup" not in runs[0].args


def test_committed_night2_capped_lm():
    spec = load_spec(_SPEC_DIR / "night2_capped_lm.toml")
    runs = expand(spec)
    assert len(runs) == 3
    methods = {r.args["method"] for r in runs}
    assert methods == {"das_true", "joint", "random_rotation"}
    assert runs[0].args["sparse_mode"] == "per_dim"
    assert runs[0].args["max_width"] == 128
