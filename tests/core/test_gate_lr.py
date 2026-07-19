"""Tests for gate optimization knobs (gate_lr / gate_init).

Regression coverage for the Adam-speed-limit failure mode: with the default
lr=1e-3 the gate log_alpha cannot travel from its +2.0 init down past the ~-0.4
live threshold within a few hundred steps, so lambda_gate is inert. A dedicated
gate_lr fixes this. Covers:

- optimizer param groups: gates get their own lr when gate_lr is set, and fall
  back to the shared lr otherwise;
- gate_init below the live threshold yields a dead g_det at init;
- a short synthetic training loop actually closes gates with gate_lr=0.05 /
  lambda_gate=1.0, and does NOT with gate_lr=None (lr=1e-3).
"""

from __future__ import annotations

import torch

from jdas.causal_model import LearnedCausalModel
from jdas.gates import VariableGates
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import JointConfig, JointTrainer

from .fakes import MLPSite, XorTask


def _build_trainer(
    *,
    steps: int,
    lambda_gate: float,
    gate_lr: float | None,
    gate_init: float = 2.0,
    k_max: int = 4,
    d: int = 8,
) -> JointTrainer:
    """A tiny gated JointTrainer on the XOR fake task (runs in ms on CPU)."""
    torch.manual_seed(0)
    task = XorTask(emb=2)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=d, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=k_max, v=2, n_labels=2)
    rot = OrthogonalRotation(d)
    layout = SubspaceLayout(d, k_max, init_width=1.0)
    gates = VariableGates(k_max, init=gate_init)
    cfg = JointConfig(
        steps=steps,
        batch_size=32,
        n_sources=2,
        lr=1e-3,
        eval_every=0,
        seed=0,
        device="cpu",
        use_gates=True,
        lambda_gate=lambda_gate,
        gate_lr=gate_lr,
        gate_init=gate_init,
    )
    return JointTrainer(site, task, causal, rot, layout, cfg, gates=gates)


def _run_loop(trainer: JointTrainer) -> None:
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch)
        trainer.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        trainer.optimizer.step()


# ---------------------------------------------------------------------------
# optimizer param groups
# ---------------------------------------------------------------------------


def _gate_group(trainer: JointTrainer) -> dict:
    """The optimizer param group that owns the gate log_alpha parameter."""
    gate_param = trainer.gates.log_alpha
    for group in trainer.optimizer.param_groups:
        if any(p is gate_param for p in group["params"]):
            return group
    raise AssertionError("gate param not found in any optimizer group")


def test_gate_lr_creates_separate_group_with_its_lr() -> None:
    """gate_lr puts gates in their own group with that lr; others keep config.lr."""
    trainer = _build_trainer(steps=1, lambda_gate=0.1, gate_lr=0.05)
    group = _gate_group(trainer)
    assert group["lr"] == 0.05
    # Every non-gate group uses the shared lr (1e-3 here).
    gate_param = trainer.gates.log_alpha
    for g in trainer.optimizer.param_groups:
        if not any(p is gate_param for p in g["params"]):
            assert g["lr"] == trainer.config.lr


def test_gate_lr_none_falls_back_to_config_lr() -> None:
    """With gate_lr=None the gate group's effective lr is config.lr."""
    trainer = _build_trainer(steps=1, lambda_gate=0.1, gate_lr=None)
    group = _gate_group(trainer)
    assert group["lr"] == trainer.config.lr


# ---------------------------------------------------------------------------
# gate_init below the live threshold
# ---------------------------------------------------------------------------


def test_gate_init_negative_starts_dead() -> None:
    """VariableGates(init=-1.0) yields g_det <= 0.5 (dead) at init."""
    g = VariableGates(4, init=-1.0)
    det = g.deterministic()
    assert bool((det <= 0.5).all())
    assert g.gated_k() == 0
    # The default init is live for contrast.
    live = VariableGates(4, init=2.0)
    assert bool((live.deterministic() > 0.5).all())
    assert live.gated_k() == 4


# ---------------------------------------------------------------------------
# regression: gate_lr actually closes gates, lr=1e-3 does not
# ---------------------------------------------------------------------------


def test_gate_lr_closes_gates_where_default_lr_does_not() -> None:
    """The tonight-bug regression.

    With lambda_gate=1.0 and a dedicated gate_lr=0.05 a few hundred steps close
    gates (gated_k drops below k_max). With gate_lr=None (so the gate lr is the
    shared 1e-3) the same penalty over the same steps cannot move log_alpha far
    enough, so gated_k stays at k_max.
    """
    k_max = 4
    steps = 400

    fast = _build_trainer(
        steps=steps, lambda_gate=1.0, gate_lr=0.05, k_max=k_max
    )
    assert fast.gates.gated_k() == k_max  # all live at init (+2.0)
    _run_loop(fast)
    gated_k_fast = fast.gates.gated_k()

    slow = _build_trainer(
        steps=steps, lambda_gate=1.0, gate_lr=None, k_max=k_max
    )
    assert slow.gates.gated_k() == k_max
    _run_loop(slow)
    gated_k_slow = slow.gates.gated_k()

    assert gated_k_fast < k_max, (
        f"gate_lr=0.05 should close at least one gate, got gated_k={gated_k_fast}"
    )
    assert gated_k_slow == k_max, (
        f"gate_lr=None (lr=1e-3) should close no gate, got gated_k={gated_k_slow}"
    )
