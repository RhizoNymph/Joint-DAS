"""Phase A entry point (thin shim).

The run logic now lives in :mod:`jdas.cli.runners`; the canonical invocation is
``jdas run phase-a ...``.  This shim keeps ``python experiments/run_phase_a.py``
working with byte-identical CLI behavior and re-exports the helpers that
existing tests and committed result ``config`` blocks depend on.

Methods
-------
- ``joint``          -- learn Q, boundaries, and H jointly (ours).
- ``das_true``       -- classic DAS with the true hand-specified H (upper bound).
- ``das_wrong``      -- classic DAS with a wrong H (single output-copy variable).
- ``das_wrong_and``  -- classic DAS with the TRUE ground-truth variables but a
  WRONG composition law (k=2), the principled falsification baseline.
- ``random_rotation``-- joint H but Q frozen at random init (control).
"""

from __future__ import annotations

from jdas.cli.runners import (
    _GATE_METHODS,
    _add_recovery,
    _build_config_a as _build_config,
    _fixed_swap_assignment,
    _infer_input_dim,
    _load_site_a as _load_site,
    _load_task_a as _load_task,
    _make_layout,
    _true_fixed_model,
    _validate_gate_method,
    _wrong_and_agreement_ceiling,
    _wrong_and_fixed_model,
    _wrong_fixed_model,
    _wrong_law_label_fn,
    build_phase_a_parser as _build_parser,
    run_phase_a,
)

__all__ = [
    "_GATE_METHODS",
    "_add_recovery",
    "_build_config",
    "_build_parser",
    "_fixed_swap_assignment",
    "_infer_input_dim",
    "_load_site",
    "_load_task",
    "_make_layout",
    "_true_fixed_model",
    "_validate_gate_method",
    "_wrong_and_agreement_ceiling",
    "_wrong_and_fixed_model",
    "_wrong_fixed_model",
    "_wrong_law_label_fn",
    "main",
]


def main() -> None:
    args = _build_parser().parse_args()
    run_phase_a(args)


if __name__ == "__main__":
    main()
