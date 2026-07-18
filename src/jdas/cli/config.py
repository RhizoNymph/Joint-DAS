"""Environment configuration for the ``jdas`` CLI.

Loads a checked-in, non-secret TOML file (``jdas.toml``) describing this
machine's cluster hosts, remote paths, uv location, HF env vars, result/log
paths, and default model ids into a frozen typed :class:`EnvConfig`.

Precedence (highest first): ``--config PATH`` flag > ``JDAS_CONFIG`` env var >
``./jdas.local.toml`` (gitignored) > ``./jdas.toml``.  A missing config file
yields the built-in defaults with a warning so the CLI works out of the box.

Unknown keys are a hard error (typo guard): the config is the single source of
truth for host/path/model specifics, and a silently-ignored typo would send a
job to the wrong place.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (mirror docs/features/unified-cli.md; this repo's jdas.toml matches).
# ---------------------------------------------------------------------------

_DEFAULT_HOSTS: tuple[str, ...] = ("node0", "node1", "node2")
_DEFAULT_REMOTE_DIR = "Code/ai/learning-causal-representations"
_DEFAULT_UV_PATH = "~/.local/bin/uv"
_DEFAULT_CLUSTER_ENV: dict[str, str] = {
    "HF_HOME": "$HOME/hf-cache",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_XET": "1",
}
_DEFAULT_PATHS: dict[str, str] = {
    "results": "experiments/results",
    "logs": "experiments/logs",
    "ckpts": "experiments/ckpts",
    "toy_ckpts": "experiments/toy_ckpts",
}
_DEFAULT_MODELS: dict[str, str] = {
    "lm_default": "Qwen/Qwen2.5-1.5B-Instruct",
}


class ConfigError(Exception):
    """Raised on a malformed config file or an unknown/typo'd key."""


@dataclass(frozen=True)
class ClusterConfig:
    """SSH cluster settings: hosts, remote layout, uv path, exported env."""

    hosts: tuple[str, ...] = _DEFAULT_HOSTS
    remote_dir: str = _DEFAULT_REMOTE_DIR
    uv_path: str = _DEFAULT_UV_PATH
    env: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_CLUSTER_ENV))

    def env_exports(self) -> str:
        """``export K=V K=V ...`` line for prefixing remote commands.

        Deterministic order (sorted keys) so generated drivers are stable.
        """
        if not self.env:
            return ""
        parts = " ".join(f"{k}={self.env[k]}" for k in sorted(self.env))
        return f"export {parts}"


@dataclass(frozen=True)
class PathsConfig:
    """Result/log/checkpoint directories, all relative to the repo root."""

    results: str = _DEFAULT_PATHS["results"]
    logs: str = _DEFAULT_PATHS["logs"]
    ckpts: str = _DEFAULT_PATHS["ckpts"]
    toy_ckpts: str = _DEFAULT_PATHS["toy_ckpts"]


@dataclass(frozen=True)
class ModelsConfig:
    """Default model ids referenced by runners/sweeps."""

    lm_default: str = _DEFAULT_MODELS["lm_default"]


@dataclass(frozen=True)
class EnvConfig:
    """Top-level environment config loaded from ``jdas.toml``."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    source: str = "<defaults>"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _reject_unknown(table: dict, allowed: set[str], where: str) -> None:
    """Raise :class:`ConfigError` if ``table`` has keys outside ``allowed``."""
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in [{where}]; "
            f"allowed: {sorted(allowed)}"
        )


def _dataclass_keys(cls) -> set[str]:
    return {f.name for f in fields(cls) if f.name != "source"}


def _build_cluster(table: dict) -> ClusterConfig:
    _reject_unknown(table, _dataclass_keys(ClusterConfig), "cluster")
    base = ClusterConfig()
    kwargs: dict = {}
    if "hosts" in table:
        kwargs["hosts"] = tuple(str(h) for h in table["hosts"])
    if "remote_dir" in table:
        kwargs["remote_dir"] = str(table["remote_dir"])
    if "uv_path" in table:
        kwargs["uv_path"] = str(table["uv_path"])
    if "env" in table:
        env = table["env"]
        if not isinstance(env, dict):
            raise ConfigError("[cluster.env] must be a table of string=string")
        kwargs["env"] = {str(k): str(v) for k, v in env.items()}
    return replace(base, **kwargs)


def _build_paths(table: dict) -> PathsConfig:
    _reject_unknown(table, _dataclass_keys(PathsConfig), "paths")
    return replace(PathsConfig(), **{k: str(v) for k, v in table.items()})


def _build_models(table: dict) -> ModelsConfig:
    _reject_unknown(table, _dataclass_keys(ModelsConfig), "models")
    return replace(ModelsConfig(), **{k: str(v) for k, v in table.items()})


def config_from_dict(data: dict, *, source: str = "<dict>") -> EnvConfig:
    """Build an :class:`EnvConfig` from a parsed TOML mapping.

    Top-level and per-section unknown keys are errors.
    """
    _reject_unknown(data, {"cluster", "paths", "models"}, "root")
    return EnvConfig(
        cluster=_build_cluster(data.get("cluster", {})),
        paths=_build_paths(data.get("paths", {})),
        models=_build_models(data.get("models", {})),
        source=source,
    )


def _resolve_path(explicit: str | os.PathLike | None) -> Path | None:
    """Apply precedence to find the config file, or None to use defaults.

    ``--config`` > ``JDAS_CONFIG`` > ``jdas.local.toml`` > ``jdas.toml``.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("JDAS_CONFIG")
    if env:
        return Path(env)
    for name in ("jdas.local.toml", "jdas.toml"):
        candidate = Path(name)
        if candidate.exists():
            return candidate
    return None


def load_config(
    explicit: str | os.PathLike | None = None,
    *,
    warn: bool = True,
) -> EnvConfig:
    """Load the environment config, applying precedence.

    A ``--config``/``JDAS_CONFIG`` path that does not exist is an error; a
    missing default file yields built-in defaults (with a warning to stderr).
    """
    path = _resolve_path(explicit)
    if path is None:
        if warn:
            print(
                "warning: no jdas.toml found; using built-in defaults",
                file=sys.stderr,
            )
        return EnvConfig()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"malformed TOML in {path}: {exc}") from exc
    return config_from_dict(data, source=str(path))
