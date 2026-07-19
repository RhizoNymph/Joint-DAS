"""``jdas analyze`` dispatch to the analyzer modules.

Thin adapters that build an argparse for each analyzer subcommand (so default
paths and flags stay stable) and call the corresponding module entry point with
a rebuilt ``sys.argv``.  The analyzers live in ``experiments/``:

- ``analyze_gates``   -> ``jdas analyze gates`` (night-3 gate sweeps).
- ``analyze_toy_lm``  -> ``jdas analyze toy`` (toy-model / LM aggregate table).
- ``analyze_studies`` -> ``jdas analyze capped-lm | seed-basis | search |
  falsification`` (one descriptive subcommand per study plot).

The analyzers live under ``experiments/`` (a namespace package that is not
installed), so we make the repo root importable before importing them — the
installed ``jdas`` console script does not otherwise have the repo root on
``sys.path`` the way ``python experiments/analyze_gates.py`` does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root = three levels up from this file (src/jdas/cli/analyze.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _ensure_experiments_importable() -> None:
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def build_gates_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Night-3 gate-sweep analyzer")
    p.add_argument("--toy-dir", default="experiments/results/night3/gates_toy")
    p.add_argument("--lm-dir", default="experiments/results/night3/gates_lm")
    p.add_argument("--out-md", default="experiments/results/night3/gates_summary.md")
    p.add_argument("--plot", default="docs/assets/night3_gates.png")
    return p


def build_toy_analyze_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Aggregate toy-model / LM results")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--assets-dir", default="docs/assets")
    p.add_argument("--tag", default="phase_a")
    return p


def build_studies_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Joint-DAS study plot")
    # Default points at the historical committed data directory (unchanged).
    p.add_argument("--studies-dir", default="experiments/results/night2")
    p.add_argument("--assets-dir", default="docs/assets")
    return p


def _call_main(module_name: str, main_attr: str, argv: list[str]) -> None:
    """Import ``module_name`` and invoke its ``main()`` with a rebuilt argv."""
    _ensure_experiments_importable()
    import importlib

    module = importlib.import_module(module_name)
    saved = sys.argv
    sys.argv = [module_name, *argv]
    try:
        getattr(module, main_attr)()
    finally:
        sys.argv = saved


def run_gates(argv: list[str]) -> None:
    _call_main("experiments.analyze_gates", "main", argv)


def run_toy_analyze(argv: list[str]) -> None:
    _call_main("experiments.analyze_toy_lm", "main", argv)


def run_capped_lm(argv: list[str]) -> None:
    _call_main("experiments.analyze_studies", "capped_lm", argv)


def run_seed_basis(argv: list[str]) -> None:
    _call_main("experiments.analyze_studies", "seed_basis", argv)


def run_search_ranking(argv: list[str]) -> None:
    _call_main("experiments.analyze_studies", "search_ranking", argv)


def run_falsification(argv: list[str]) -> None:
    _call_main("experiments.analyze_studies", "falsification", argv)
