"""Phase B entry point (thin shim).

The run logic now lives in :mod:`jdas.cli.runners`; the canonical invocation is
``jdas run phase-b ...``.  This shim keeps ``python experiments/run_phase_b.py``
working with byte-identical CLI behavior and re-exports the helpers tests and
committed result ``config`` blocks depend on.

The intervention site is the residual stream at the last token of one decoder
layer of a frozen HF causal LM; the learned causal model reads decoded
``(X, Y, Z)`` features from the token ids.
"""

from __future__ import annotations

from jdas.cli.runners import (
    _GATE_METHODS,
    _add_recovery_b as _add_recovery,
    _build_config_b as _build_config,
    _maybe_save_ckpt_b as _maybe_save_ckpt,
    _true_fixed_model_b as _true_fixed_model,
    _validate_gate_method,
    _wrong_fixed_model_b as _wrong_fixed_model,
    build_phase_b_parser as _build_parser,
    run_phase_b,
)

__all__ = [
    "_GATE_METHODS",
    "_add_recovery",
    "_build_config",
    "_build_parser",
    "_maybe_save_ckpt",
    "_true_fixed_model",
    "_validate_gate_method",
    "_wrong_fixed_model",
    "main",
]


def main() -> None:
    args = _build_parser().parse_args()
    run_phase_b(args)


if __name__ == "__main__":
    main()
