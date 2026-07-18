"""Declarative sweep specs: grid expansion, out-pattern rendering, execution.

A sweep spec is a TOML file with ``[sweep]``, ``[grid]``, and ``[fixed]``
tables.  The grid is a cartesian product over its axes; an axis named ``"a+b"``
zips several parameters that vary together (each grid point supplies a tuple).
Each expanded run is a mapping of CLI-argument names to values plus a rendered
output path.

Expansion is deterministic: same spec -> same ordered run list -> same
round-robin host assignment (``host = index % len(hosts)`` over the run list in
spec order).  A run whose output JSON already exists is skipped (idempotent
resume).

The local executor shells out to ``uv run jdas run <runner> ...`` per run; the
cluster executor generates one bash driver per host (see
:func:`render_driver`) which :mod:`jdas.cli.cluster` copies over and launches
detached.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from .config import EnvConfig

# Map long task names to the short forms used in out-pattern ``{task_short}``.
_TASK_SHORT = {
    "hierarchical_equality": "hier",
    "boolean_comp": "bool",
    "price_tagging": "pt",
}

# Sweep runner -> the ``jdas run <subcommand>`` it maps to.
_RUNNERS = ("toy", "lm", "search", "seed-study")


class SweepError(Exception):
    """Raised on a malformed sweep spec."""


@dataclass(frozen=True)
class SweepSpec:
    """A parsed sweep spec (``[sweep]``/``[grid]``/``[fixed]``)."""

    name: str
    runner: str
    out_dir: str
    out_pattern: str
    grid: dict[str, list]
    fixed: dict[str, object] = field(default_factory=dict)
    source: str = "<dict>"


@dataclass(frozen=True)
class Run:
    """One expanded run: argument mapping + output path (relative to results)."""

    index: int
    args: dict[str, object]
    out_rel: str  # out_dir/rendered_filename, under paths.results

    def out_path(self, cfg: EnvConfig) -> Path:
        return Path(cfg.paths.results) / self.out_rel


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def spec_from_dict(data: dict, *, source: str = "<dict>") -> SweepSpec:
    """Validate and build a :class:`SweepSpec` from a parsed TOML mapping."""
    if "sweep" not in data:
        raise SweepError("missing [sweep] table")
    sweep = data["sweep"]
    for key in ("name", "runner", "out_dir", "out_pattern"):
        if key not in sweep:
            raise SweepError(f"[sweep] missing required key {key!r}")
    runner = str(sweep["runner"])
    if runner not in _RUNNERS:
        raise SweepError(f"[sweep].runner {runner!r} not in {_RUNNERS}")
    grid = data.get("grid", {})
    if not isinstance(grid, dict):
        raise SweepError("[grid] must be a table")
    fixed = data.get("fixed", {})
    if not isinstance(fixed, dict):
        raise SweepError("[fixed] must be a table")
    # Validate zipped axes: every list entry must be a same-length tuple.
    for axis, values in grid.items():
        if not isinstance(values, list):
            raise SweepError(f"[grid].{axis} must be a list")
        if "+" in axis:
            names = axis.split("+")
            for entry in values:
                if not isinstance(entry, list) or len(entry) != len(names):
                    raise SweepError(
                        f"[grid].{axis!r} zipped axis needs {len(names)}-tuples; "
                        f"got {entry!r}"
                    )
    return SweepSpec(
        name=str(sweep["name"]),
        runner=runner,
        out_dir=str(sweep["out_dir"]),
        out_pattern=str(sweep["out_pattern"]),
        grid={str(k): list(v) for k, v in grid.items()},
        fixed=dict(fixed),
        source=source,
    )


def load_spec(path: str | Path) -> SweepSpec:
    """Load and validate a sweep spec TOML file."""
    p = Path(path)
    if not p.exists():
        raise SweepError(f"sweep spec not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    return spec_from_dict(data, source=str(p))


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _axis_points(axis: str, values: list) -> list[dict]:
    """Expand one axis into a list of ``{arg_name: value}`` partial mappings."""
    if "+" in axis:
        names = axis.split("+")
        return [dict(zip(names, entry, strict=True)) for entry in values]
    return [{axis: v} for v in values]


def expand(spec: SweepSpec) -> list[Run]:
    """Deterministically expand the grid into an ordered list of runs.

    Cartesian product over axes in spec (insertion) order; fixed args merged
    into every run; output path rendered from ``out_pattern``.
    """
    axes = list(spec.grid.items())
    per_axis = [_axis_points(axis, values) for axis, values in axes]
    runs: list[Run] = []
    for index, combo in enumerate(product(*per_axis) if per_axis else [()]):
        args: dict[str, object] = dict(spec.fixed)
        for partial in combo:
            args.update(partial)
        filename = render_out(spec.out_pattern, args)
        out_rel = str(Path(spec.out_dir) / filename)
        runs.append(Run(index=index, args=args, out_rel=out_rel))
    return runs


def render_out(pattern: str, args: dict) -> str:
    """Render an out-pattern.  Supports ``{arg}`` and the derived
    ``{task_short}`` (from ``args['task']``)."""
    fields: dict[str, object] = dict(args)
    task = args.get("task")
    if task is not None:
        fields["task_short"] = _TASK_SHORT.get(str(task), str(task))
    try:
        return pattern.format(**fields)
    except KeyError as exc:
        raise SweepError(
            f"out_pattern {pattern!r} references {exc} not present in run args "
            f"{sorted(fields)}"
        ) from exc


def assign_hosts(runs: list[Run], hosts: tuple[str, ...]) -> dict[str, list[Run]]:
    """Round-robin ``runs`` across ``hosts`` by ``index % len(hosts)``."""
    if not hosts:
        raise SweepError("no cluster hosts configured")
    buckets: dict[str, list[Run]] = {h: [] for h in hosts}
    for run in runs:
        buckets[hosts[run.index % len(hosts)]].append(run)
    return buckets


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _flagify(name: str) -> str:
    """``site_layer`` -> ``--site-layer``."""
    return "--" + name.replace("_", "-")


def run_argv(spec: SweepSpec, run: Run, out_path: str) -> list[str]:
    """Argument vector (after ``jdas run <runner>``) for one run.

    Boolean ``True`` flags become bare ``--flag``; ``False``/``None`` are
    dropped; everything else is ``--flag value``.  ``--out`` is appended last.
    """
    argv: list[str] = ["run", spec.runner]
    for name, value in run.args.items():
        flag = _flagify(name)
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    argv.extend(["--out", out_path])
    return argv


def render_driver(
    spec: SweepSpec,
    runs: list[Run],
    cfg: EnvConfig,
    host: str,
) -> str:
    """Generate the bash driver a host runs for its share of the sweep.

    Encodes the hard-won remote rules: cd into ``remote_dir``, export the
    cluster env, per-run logs under ``paths.logs``, a ``<name>_failures.log``
    line on failure, and a ``DONE_<host>`` line to ``<name>_status.log`` when
    the queue drains.  Runs whose output already exists are skipped at run time
    (``[ -f ... ]``) as a second guard on top of local skip-existing.
    """
    logs = cfg.paths.logs
    results = cfg.paths.results
    status_log = f"{logs}/{spec.name}_status.log"
    fail_log = f"{logs}/{spec.name}_failures.log"
    lines = [
        "#!/usr/bin/env bash",
        f"# generated driver for sweep {spec.name!r} on host {host}",
        f"cd $HOME/{cfg.cluster.remote_dir}",
    ]
    env_line = cfg.cluster.env_exports()
    if env_line:
        lines.append(env_line)
    # Ensure output + log dirs exist.
    out_dirs = sorted({str(Path(results) / spec.out_dir)})
    lines.append(f"mkdir -p {' '.join(shlex.quote(d) for d in out_dirs)} {shlex.quote(logs)}")
    for run in runs:
        out_path = str(Path(results) / run.out_rel)
        log_path = f"{logs}/{spec.name}_{Path(run.out_rel).stem}.log"
        argv = run_argv(spec, run, out_path)
        cmd = f"{cfg.cluster.uv_path} run jdas " + " ".join(
            shlex.quote(a) for a in argv
        )
        tag = Path(run.out_rel).stem
        lines.append(f"if [ -f {shlex.quote(out_path)} ]; then")
        lines.append(f"  echo 'skip {tag} (exists)'")
        lines.append("else")
        lines.append(
            f"  {cmd} > {shlex.quote(log_path)} 2>&1 "
            f"|| echo 'FAILED {tag}' >> {shlex.quote(fail_log)}"
        )
        lines.append("fi")
    lines.append(f"echo 'DONE_{host}' >> {shlex.quote(status_log)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Local executor
# ---------------------------------------------------------------------------


def pending_runs(runs: list[Run], cfg: EnvConfig) -> list[Run]:
    """Runs whose output JSON does not yet exist (idempotent resume)."""
    return [r for r in runs if not r.out_path(cfg).exists()]


def run_local(
    spec: SweepSpec,
    runs: list[Run],
    cfg: EnvConfig,
    *,
    parallel: int = 1,
) -> int:
    """Execute pending runs locally via ``uv run jdas run ...``.

    Sequential by default; ``parallel>1`` runs up to N concurrently.  Returns
    the number of runs that failed (non-zero exit).
    """
    pending = pending_runs(runs, cfg)
    logs_dir = Path(cfg.paths.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    if parallel <= 1:
        for run in pending:
            failures += _run_one_local(spec, run, cfg)
        return failures
    # Bounded concurrency.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = pool.map(lambda r: _run_one_local(spec, r, cfg), pending)
    return sum(results)


def _run_one_local(spec: SweepSpec, run: Run, cfg: EnvConfig) -> int:
    out_path = str(run.out_path(cfg))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    argv = [cfg.cluster.uv_path, "run", "jdas", *run_argv(spec, run, out_path)]
    # uv_path may be ``~/.local/bin/uv``; expand for local exec.
    argv[0] = str(Path(argv[0]).expanduser())
    log_path = Path(cfg.paths.logs) / f"{spec.name}_{Path(run.out_rel).stem}.log"
    print(f"[local] {Path(run.out_rel).stem}")
    with log_path.open("w") as log:
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  FAILED (see {log_path})", file=sys.stderr)
        return 1
    return 0
