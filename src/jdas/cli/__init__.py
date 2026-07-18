"""``jdas`` unified CLI: single entry point for runs, sweeps, cluster ops, analysis.

Command tree (see docs/features/unified-cli.md):

    jdas run   phase-a | phase-b | search | seed-study   [runner args]
    jdas analyze gates | phase-a | night2                [analyzer args]
    jdas sweep run    SPEC.toml [--where local|cluster] [--wait] [--dry-run] [--parallel N]
    jdas sweep status SPEC.toml
    jdas sweep collect SPEC.toml
    jdas cluster sync | status | exec -- CMD... | kill PATTERN

A global ``--config PATH`` selects the environment config (precedence:
``--config`` > ``JDAS_CONFIG`` > ``jdas.local.toml`` > ``jdas.toml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analyze, cluster, runners, sweeps
from .config import ConfigError, EnvConfig, load_config
from .sweeps import SweepError


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jdas", description="Joint-DAS unified CLI")
    parser.add_argument(
        "--config",
        default=None,
        help="path to an env config TOML (overrides JDAS_CONFIG / jdas.toml)",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    _build_run_group(sub)
    _build_analyze_group(sub)
    _build_sweep_group(sub)
    _build_cluster_group(sub)
    return parser


def _build_run_group(sub) -> None:
    run = sub.add_parser("run", help="single experiment run")
    run_sub = run.add_subparsers(dest="runner", required=True)
    # Reuse each runner's own parser as the subcommand parser.
    _add_child(run_sub, "phase-a", runners.build_phase_a_parser, runners.run_phase_a)
    _add_child(run_sub, "phase-b", runners.build_phase_b_parser, runners.run_phase_b)
    _add_child(run_sub, "search", runners.build_search_parser, runners.run_search)
    _add_child(run_sub, "seed-study", runners.build_seed_study_parser, runners.run_seed_study)


def _add_child(sub, name: str, build, run_fn) -> None:
    """Attach a runner's arguments to a subparser and record its run function."""
    child = build(prog=f"jdas run {name}")
    added = sub.add_parser(
        name, parents=[child], add_help=False, description=child.description
    )
    added.set_defaults(_run_fn=run_fn)


def _build_analyze_group(sub) -> None:
    an = sub.add_parser("analyze", help="aggregate result JSONs into tables/plots")
    an_sub = an.add_subparsers(dest="analyzer", required=True)
    _add_analyze(an_sub, "gates", analyze.build_gates_parser, analyze.run_gates)
    _add_analyze(an_sub, "phase-a", analyze.build_phase_a_analyze_parser, analyze.run_phase_a_analyze)
    _add_analyze(an_sub, "night2", analyze.build_night2_parser, analyze.run_night2)


def _add_analyze(sub, name: str, build, run_fn) -> None:
    child = build(prog=f"jdas analyze {name}")
    added = sub.add_parser(
        name, parents=[child], add_help=False, description=child.description
    )
    added.set_defaults(_analyze_fn=run_fn)


def _build_sweep_group(sub) -> None:
    sw = sub.add_parser("sweep", help="declarative sweep specs")
    sw_sub = sw.add_subparsers(dest="sweep_cmd", required=True)

    run = sw_sub.add_parser("run", help="expand + execute a sweep spec")
    run.add_argument("spec")
    run.add_argument("--where", choices=["local", "cluster"], default="local")
    run.add_argument("--parallel", type=int, default=1, help="local concurrency")
    run.add_argument("--wait", action="store_true", help="poll cluster status until done")
    run.add_argument("--poll", type=int, default=300, help="--wait poll interval (s)")
    run.add_argument("--dry-run", action="store_true", help="print run list + host assignment")

    status = sw_sub.add_parser("status", help="expected vs present outputs + failures")
    status.add_argument("spec")

    collect = sw_sub.add_parser("collect", help="rsync results back from every host")
    collect.add_argument("spec")


def _build_cluster_group(sub) -> None:
    cl = sub.add_parser("cluster", help="ssh cluster operations")
    cl_sub = cl.add_subparsers(dest="cluster_cmd", required=True)
    cl_sub.add_parser("sync", help="rsync repo + uv sync on every host")
    cl_sub.add_parser("status", help="per host: relevant processes + GPU memory")
    ex = cl_sub.add_parser("exec", help="run a command on every host")
    ex.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD...")
    kill = cl_sub.add_parser("kill", help="pkill -f PATTERN on every host")
    kill.add_argument("pattern")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch_sweep(args: argparse.Namespace, cfg: EnvConfig) -> int:
    spec = sweeps.load_spec(args.spec)
    runs = sweeps.expand(spec)
    if args.sweep_cmd == "run":
        if args.dry_run:
            _print_dry_run(spec, runs, cfg, args.where)
            return 0
        if args.where == "local":
            return sweeps.run_local(spec, runs, cfg, parallel=args.parallel)
        return _run_cluster(spec, runs, cfg, wait=args.wait, poll=args.poll)
    if args.sweep_cmd == "status":
        return _sweep_status(spec, runs, cfg)
    if args.sweep_cmd == "collect":
        for host in cfg.cluster.hosts:
            print(f"=== collect {spec.out_dir} from {host} ===")
            cluster._run(cluster.rsync_collect_argv(cfg, host, spec.out_dir))
        return 0
    raise SweepError(f"unknown sweep command {args.sweep_cmd!r}")


def _print_dry_run(spec, runs, cfg: EnvConfig, where: str) -> None:
    pending = sweeps.pending_runs(runs, cfg)
    print(f"sweep {spec.name!r}: {len(runs)} runs ({len(pending)} pending) runner={spec.runner}")
    for run in runs:
        exists = "" if run.out_path(cfg).exists() else " [pending]"
        print(f"  [{run.index:>3}] {run.out_rel}{exists}")
    if where == "cluster":
        buckets = sweeps.assign_hosts(runs, cfg.cluster.hosts)
        print("host assignment (round-robin):")
        for host in cfg.cluster.hosts:
            print(f"  {host}: {len(buckets[host])} runs")
            for run in buckets[host]:
                print(f"    - {Path(run.out_rel).name}")


def _sweep_status(spec, runs, cfg: EnvConfig) -> int:
    present = sum(1 for r in runs if r.out_path(cfg).exists())
    print(f"sweep {spec.name!r}: {present}/{len(runs)} outputs present (local)")
    fail_log = Path(cfg.paths.logs) / f"{spec.name}_failures.log"
    if fail_log.exists():
        text = fail_log.read_text().strip()
        if text:
            print(f"failures ({fail_log}):")
            print(text)
    status_log = Path(cfg.paths.logs) / f"{spec.name}_status.log"
    if status_log.exists():
        print(f"status log ({status_log}):")
        print(status_log.read_text().strip())
    return 0


def _run_cluster(spec, runs, cfg: EnvConfig, *, wait: bool, poll: int) -> int:
    import tempfile
    import time

    buckets = sweeps.assign_hosts(runs, cfg.cluster.hosts)
    for host, host_runs in buckets.items():
        if not host_runs:
            continue
        driver = sweeps.render_driver(spec, host_runs, cfg, host)
        remote_dir = f"{cfg.cluster.remote_dir}/{cfg.paths.logs}"
        remote_path = f"{cfg.cluster.remote_dir}/{cfg.paths.logs}/{spec.name}_{host}.sh"
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(driver)
            local_path = fh.name
        cluster._run(cluster.ssh_argv(host, f"mkdir -p {remote_dir}"))
        cluster._run(["scp", "-q", local_path, f"{host}:{remote_path}"])
        cluster._run(cluster.launch_driver_cmd(host, remote_path))
        print(f"launched {len(host_runs)} runs on {host}")
    if wait:
        while True:
            done = sum(1 for r in runs if r.out_path(cfg).exists())
            print(f"[wait] {done}/{len(runs)} outputs present")
            if done >= len(runs):
                break
            time.sleep(poll)
        for host in cfg.cluster.hosts:
            cluster._run(cluster.rsync_collect_argv(cfg, host, spec.out_dir))
    return 0


def _dispatch_cluster(args: argparse.Namespace, cfg: EnvConfig) -> int:
    if args.cluster_cmd == "sync":
        return cluster.do_sync(cfg)
    if args.cluster_cmd == "status":
        cluster.do_status(cfg)
        return 0
    if args.cluster_cmd == "exec":
        argv = [a for a in args.cmd if a != "--"]
        return cluster.do_exec(cfg, argv)
    if args.cluster_cmd == "kill":
        cluster.do_kill(cfg, args.pattern)
        return 0
    raise SystemExit(f"unknown cluster command {args.cluster_cmd!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.group == "run":
            run_fn = args._run_fn
            run_fn(_runner_namespace(args))
            return 0
        if args.group == "analyze":
            # Re-derive the analyzer's own argv (its parser was merged in).
            return _dispatch_analyze(args, argv)
        if args.group == "sweep":
            return _dispatch_sweep(args, cfg)
        if args.group == "cluster":
            return _dispatch_cluster(args, cfg)
    except (SweepError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise SystemExit(f"unknown command group {args.group!r}")


def _runner_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Strip CLI-tree bookkeeping (``group``, ``runner``, ``_run_fn``, ...) so
    the runner sees only its own arguments — the runners serialize
    ``vars(args)`` into the result ``config`` block, which must stay clean and
    identical to the standalone-script schema."""
    drop = {"group", "runner", "analyzer", "config", "_run_fn", "_analyze_fn"}
    return argparse.Namespace(
        **{k: v for k, v in vars(args).items() if k not in drop and not k.startswith("_")}
    )


def _dispatch_analyze(args: argparse.Namespace, argv: list[str] | None) -> int:
    """Rebuild the analyzer's argv from the parsed namespace and call it."""
    fn = args._analyze_fn
    name = args.analyzer
    if name == "gates":
        passthru = _rebuild_argv(
            args, [("--toy-dir", "toy_dir"), ("--lm-dir", "lm_dir"),
                   ("--out-md", "out_md"), ("--plot", "plot")]
        )
    elif name == "phase-a":
        passthru = _rebuild_argv(
            args, [("--results-dir", "results_dir"), ("--out-md", "out_md"),
                   ("--assets-dir", "assets_dir"), ("--tag", "tag")]
        )
    else:  # night2
        passthru = _rebuild_argv(
            args, [("--night2-dir", "night2_dir"), ("--assets-dir", "assets_dir")]
        )
    fn(passthru)
    return 0


def _rebuild_argv(args: argparse.Namespace, mapping: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for flag, attr in mapping:
        val = getattr(args, attr, None)
        if val is not None:
            out.extend([flag, str(val)])
    return out
