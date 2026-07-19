"""Tests for the ``jdas`` environment-config loader (jdas.cli.config)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jdas.cli.config import (
    ConfigError,
    EnvConfig,
    config_from_dict,
    load_config,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# -- defaults -----------------------------------------------------------------


def test_missing_config_uses_defaults(tmp_path, monkeypatch, capsys):
    """No config file anywhere -> built-in defaults + a warning."""
    monkeypatch.delenv("JDAS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg, EnvConfig)
    assert cfg.cluster.hosts == ("node0", "node1", "node2")
    assert cfg.cluster.remote_dir == "Code/ai/learning-causal-representations"
    assert cfg.cluster.uv_path == "~/.local/bin/uv"
    assert cfg.cluster.env["HF_HOME"] == "$HOME/hf-cache"
    assert cfg.paths.results == "experiments/results"
    assert cfg.models.lm_default == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "no jdas.toml found" in capsys.readouterr().err


def test_defaults_no_warn(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("JDAS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    load_config(warn=False)
    assert capsys.readouterr().err == ""


# -- precedence ---------------------------------------------------------------


def test_precedence_explicit_over_env_and_local(tmp_path, monkeypatch):
    """--config wins over JDAS_CONFIG > jdas.local.toml > jdas.toml."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "jdas.toml", '[cluster]\nremote_dir = "from_base"\n')
    _write(tmp_path / "jdas.local.toml", '[cluster]\nremote_dir = "from_local"\n')
    envp = _write(tmp_path / "env.toml", '[cluster]\nremote_dir = "from_env"\n')
    explicit = _write(tmp_path / "explicit.toml", '[cluster]\nremote_dir = "from_explicit"\n')
    monkeypatch.setenv("JDAS_CONFIG", str(envp))

    assert load_config(str(explicit)).cluster.remote_dir == "from_explicit"


def test_precedence_env_over_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "jdas.toml", '[cluster]\nremote_dir = "from_base"\n')
    _write(tmp_path / "jdas.local.toml", '[cluster]\nremote_dir = "from_local"\n')
    envp = _write(tmp_path / "env.toml", '[cluster]\nremote_dir = "from_env"\n')
    monkeypatch.setenv("JDAS_CONFIG", str(envp))
    assert load_config().cluster.remote_dir == "from_env"


def test_precedence_local_over_base(tmp_path, monkeypatch):
    monkeypatch.delenv("JDAS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "jdas.toml", '[cluster]\nremote_dir = "from_base"\n')
    _write(tmp_path / "jdas.local.toml", '[cluster]\nremote_dir = "from_local"\n')
    assert load_config().cluster.remote_dir == "from_local"


def test_base_used_when_no_local(tmp_path, monkeypatch):
    monkeypatch.delenv("JDAS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "jdas.toml", '[cluster]\nremote_dir = "from_base"\n')
    assert load_config().cluster.remote_dir == "from_base"


def test_explicit_missing_path_is_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "nope.toml"))


# -- unknown-key guard --------------------------------------------------------


def test_unknown_top_level_key_errors():
    with pytest.raises(ConfigError):
        config_from_dict({"clustr": {}})  # typo'd section name


def test_unknown_cluster_key_errors():
    with pytest.raises(ConfigError):
        config_from_dict({"cluster": {"hostz": ["a"]}})


def test_unknown_paths_key_errors():
    with pytest.raises(ConfigError):
        config_from_dict({"paths": {"reslts": "x"}})


# -- env-var reference passthrough --------------------------------------------


def test_env_var_references_pass_through_verbatim():
    """HF env values that reference $HOME are stored verbatim (no expansion)."""
    cfg = config_from_dict(
        {"cluster": {"env": {"HF_HOME": "$HOME/hf-cache", "HF_HUB_OFFLINE": "1"}}}
    )
    assert cfg.cluster.env["HF_HOME"] == "$HOME/hf-cache"
    line = cfg.cluster.env_exports()
    assert "HF_HOME=$HOME/hf-cache" in line
    assert line.startswith("export ")
    # Deterministic (sorted) key order.
    assert cfg.cluster.env_exports() == cfg.cluster.env_exports()


def test_real_repo_jdas_toml_loads():
    """The checked-in repo jdas.toml parses and matches this environment."""
    repo_toml = Path(__file__).resolve().parents[2] / "jdas.toml"
    cfg = load_config(str(repo_toml))
    assert cfg.cluster.hosts == ("node0", "node1", "node2")
    assert cfg.cluster.uv_path == "~/.local/bin/uv"
    assert cfg.cluster.env["HF_HUB_DISABLE_XET"] == "1"
    assert cfg.models.lm_default == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.paths.ckpts == "experiments/ckpts"
