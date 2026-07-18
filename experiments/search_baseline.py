"""Discrete search baseline for Phase A (thin shim).

The logic now lives in :mod:`jdas.cli.runners`; the canonical invocation is
``jdas run search ...``.  For each (task, site) pair, enumerate every unordered
pair of distinct candidate binary variables from a small hypothesis library,
fit a majority-label lookup decoder, train the alignment with a classic
``DASTrainer``, and rank pairs by held-out ``iia_1`` / ``iia_2``.
"""

from __future__ import annotations

from jdas.cli.runners import build_search_parser as _build_parser, run_search


def main() -> None:
    args = _build_parser().parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
