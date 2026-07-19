"""Cluster primitives: ssh/rsync command construction + sync/status/exec/kill.

All subprocess/ssh construction lives here behind small pure functions that
return the argv/command string, so tests can assert on the generated commands
without ever contacting a host.  The thin ``do_*`` wrappers actually spawn
subprocesses and are exercised only interactively.

Hard-won remote rules encoded here (see docs/features/unified-cli.md):

- Detached launch uses ``ssh -f -n HOST 'nohup bash DRIVER < /dev/null ...'``.
  The ``-f`` backgrounds ssh after auth, ``-n`` + ``< /dev/null`` detach stdin;
  without them the local ssh blocks until the remote job exits.
- ``cluster kill`` wraps the pattern's first character in ``[]`` so ``pkill -f``
  can never match its own command line.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import EnvConfig

# rsync excludes shared by sync (push repo) and collect (pull results).
_SYNC_EXCLUDES = (
    ".git",
    ".venv",
    "__pycache__",
    "experiments/results",
    "experiments/logs",
    "experiments/toy_ckpts",
    "*.egg-info",
)


@dataclass(frozen=True)
class RemoteCmd:
    """A constructed command with the host it targets (for display/tests)."""

    host: str
    argv: list[str]

    def display(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


# ---------------------------------------------------------------------------
# ssh / rsync builders (pure)
# ---------------------------------------------------------------------------


def ssh_argv(host: str, remote_cmd: str, *, detach: bool = False) -> list[str]:
    """``ssh`` argv running ``remote_cmd`` on ``host``.

    ``detach=True`` adds ``-f -n`` for fire-and-forget background launches.
    """
    flags = ["-f", "-n"] if detach else ["-n"]
    return ["ssh", *flags, host, remote_cmd]


def launch_driver_cmd(host: str, driver_remote_path: str) -> list[str]:
    """Detached ssh argv that nohups a driver with stdin redirected off.

    ``ssh -f -n HOST 'nohup bash DRIVER < /dev/null > /dev/null 2>&1 &'``.
    """
    remote = (
        f"nohup bash {shlex.quote(driver_remote_path)} "
        "< /dev/null > /dev/null 2>&1 &"
    )
    return ssh_argv(host, remote, detach=True)


def rsync_push_argv(cfg: EnvConfig, host: str, repo_root: str = ".") -> list[str]:
    """rsync argv pushing the local repo to ``host:remote_dir``."""
    excludes: list[str] = []
    for pat in _SYNC_EXCLUDES:
        excludes.extend(["--exclude", pat])
    src = str(Path(repo_root)).rstrip("/") + "/"
    dst = f"{host}:{cfg.cluster.remote_dir}/"
    return ["rsync", "-az", "--delete", *excludes, src, dst]


def rsync_collect_argv(
    cfg: EnvConfig, host: str, out_rel_dir: str, repo_root: str = "."
) -> list[str]:
    """rsync argv pulling a host's ``out_rel_dir`` back into the local tree.

    ``--ignore-existing`` so already-collected results are never clobbered.
    ``out_rel_dir`` is relative to ``paths.results`` (the sweep's out_dir).
    """
    rel = str(Path(cfg.paths.results) / out_rel_dir)
    src = f"{host}:{cfg.cluster.remote_dir}/{rel}/"
    dst = str(Path(repo_root) / rel) + "/"
    return ["rsync", "-az", "--ignore-existing", src, dst]


def uv_sync_cmd(cfg: EnvConfig) -> str:
    """Remote shell command that cds into the repo and runs ``uv sync``."""
    return f"cd {cfg.cluster.remote_dir} && {cfg.cluster.uv_path} sync --quiet"


def status_cmd(cfg: EnvConfig) -> str:
    """Remote shell command reporting jdas processes + GPU memory on a host."""
    env = cfg.cluster.env_exports()
    prefix = f"{env}; " if env else ""
    # bracket-escape 'jdas run' so pgrep doesn't match itself
    return (
        f"{prefix}"
        "echo '== procs =='; "
        "pgrep -af '[j]das run' || echo '(none)'; "
        "echo '== gpu =='; "
        "nvidia-smi --query-gpu=memory.used,memory.total "
        "--format=csv,noheader 2>/dev/null || echo '(no nvidia-smi)'"
    )


def bracket_escape(pattern: str) -> str:
    """Wrap the first character of ``pattern`` in ``[]`` so ``pkill -f`` can't
    match its own command line.  ``run_phase_b`` -> ``[r]un_phase_b``."""
    if not pattern:
        return pattern
    return f"[{pattern[0]}]{pattern[1:]}"


def kill_cmd(pattern: str) -> str:
    """Remote ``pkill -f`` command with the pattern bracket-escaped."""
    return f"pkill -f {shlex.quote(bracket_escape(pattern))}"


def exec_cmd(argv: list[str]) -> str:
    """Join a ``cluster exec -- CMD...`` argv into a single remote shell string."""
    return " ".join(shlex.quote(a) for a in argv)


# ---------------------------------------------------------------------------
# Effectful wrappers (spawn subprocesses; not exercised by tests)
# ---------------------------------------------------------------------------


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True)


def do_sync(cfg: EnvConfig, repo_root: str = ".") -> int:
    """Push the repo and ``uv sync`` on every configured host."""
    failures = 0
    for host in cfg.cluster.hosts:
        print(f"=== syncing {host} ===")
        push = _run(rsync_push_argv(cfg, host, repo_root))
        if push.returncode != 0:
            print(push.stderr)
            failures += 1
            continue
        sync = _run(ssh_argv(host, uv_sync_cmd(cfg)))
        if sync.returncode != 0:
            print(sync.stderr)
            failures += 1
    return failures


def do_status(cfg: EnvConfig) -> None:
    for host in cfg.cluster.hosts:
        print(f"=== {host} ===")
        res = _run(ssh_argv(host, status_cmd(cfg)))
        print(res.stdout.rstrip() or res.stderr.rstrip())


def do_exec(cfg: EnvConfig, argv: list[str]) -> int:
    failures = 0
    cmd = exec_cmd(argv)
    for host in cfg.cluster.hosts:
        print(f"=== {host}: {cmd} ===")
        res = _run(ssh_argv(host, cmd))
        print(res.stdout.rstrip() or res.stderr.rstrip())
        failures += 1 if res.returncode != 0 else 0
    return failures


def do_kill(cfg: EnvConfig, pattern: str) -> None:
    cmd = kill_cmd(pattern)
    for host in cfg.cluster.hosts:
        print(f"=== {host}: {cmd} ===")
        _run(ssh_argv(host, cmd))
