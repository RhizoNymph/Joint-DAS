"""Tests for cluster command construction (ssh/rsync builders, kill escaping).

These assert on generated commands only; no subprocess is ever spawned.
"""

from __future__ import annotations

from dataclasses import replace

from jdas.cli.cluster import (
    bracket_escape,
    kill_cmd,
    launch_driver_cmd,
    rsync_collect_argv,
    rsync_push_argv,
    ssh_argv,
    status_cmd,
    uv_sync_cmd,
)
from jdas.cli.config import EnvConfig


def _cfg() -> EnvConfig:
    return EnvConfig()


# -- launch detach ------------------------------------------------------------


def test_launch_driver_is_detached_with_stdin_off():
    argv = launch_driver_cmd("node1", "experiments/logs/gates_v3_node1.sh")
    assert argv[0] == "ssh"
    assert "-f" in argv and "-n" in argv
    remote = argv[-1]
    assert remote.startswith("nohup bash ")
    assert "< /dev/null" in remote
    assert remote.rstrip().endswith("&")
    assert "> /dev/null 2>&1" in remote


def test_ssh_argv_non_detached_has_n_only():
    argv = ssh_argv("node0", "echo hi")
    assert argv == ["ssh", "-n", "node0", "echo hi"]


# -- rsync --------------------------------------------------------------------


def test_rsync_push_targets_remote_dir_with_excludes():
    argv = rsync_push_argv(_cfg(), "node2", repo_root=".")
    assert argv[0] == "rsync"
    assert "--delete" in argv
    assert "--exclude" in argv
    assert "experiments/results" in argv  # excluded
    assert argv[-1] == "node2:Code/ai/learning-causal-representations/"


def test_rsync_collect_ignore_existing_and_paths():
    argv = rsync_collect_argv(_cfg(), "node0", "night3/gates_toy_v3", repo_root=".")
    assert "--ignore-existing" in argv
    src = argv[-2]
    dst = argv[-1]
    assert src == "node0:Code/ai/learning-causal-representations/experiments/results/night3/gates_toy_v3/"
    assert dst.endswith("experiments/results/night3/gates_toy_v3/")


def test_uv_sync_uses_config_paths():
    cfg = replace(_cfg(), cluster=replace(_cfg().cluster, remote_dir="my/dir", uv_path="/x/uv"))
    assert uv_sync_cmd(cfg) == "cd my/dir && /x/uv sync --quiet"


# -- status -------------------------------------------------------------------


def test_status_cmd_bracket_escapes_and_exports_env():
    cmd = status_cmd(_cfg())
    assert "export HF_HOME=$HOME/hf-cache" in cmd
    assert "[j]das run" in cmd
    assert "nvidia-smi" in cmd


# -- kill bracket escaping ----------------------------------------------------


def test_bracket_escape():
    assert bracket_escape("jdas run lm") == "[j]das run lm"
    assert bracket_escape("jdas") == "[j]das"
    assert bracket_escape("") == ""


def test_kill_cmd_escaped():
    cmd = kill_cmd("jdas run lm")
    assert "[j]das run lm" in cmd
    assert cmd.startswith("pkill -f ")
    # The literal unescaped pattern must not appear (so pkill can't self-match).
    assert "'jdas run lm'" not in cmd
