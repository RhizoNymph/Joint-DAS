"""Seed / basis variance study for Phase A (thin shim).

The logic now lives in :mod:`jdas.cli.runners`; the canonical invocation is
``jdas run seed-study ...``.  For each seed, train a joint model exactly as
``jdas run phase-a`` does, then classify the learned solution (live variables,
best-matching boolean fn per variable, class in
``{atoms, equivalent_basis, output_copy, other}``) without retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable so ``experiments.*`` resolves when this
# file is run directly as ``python experiments/seed_study.py``.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jdas.cli.runners import build_seed_study_parser as _build_parser, run_seed_study


def main() -> None:
    args = _build_parser().parse_args()
    run_seed_study(args)


if __name__ == "__main__":
    main()
