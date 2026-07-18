"""Tests for the gate training schedule (warmup / lambda ramp / log_alpha clamp).

Regression coverage for the bistability failure mode (RESULTS.md N3.3): with a
matched gate_lr the gate system races to a saturated basin (all-open on the toy,
all-closed at LM scale) before the L0 penalty can act, and at both saturation
regions the penalty/sample gradient is exponentially small. The three schedule
knobs control the race:

- ``gate_warmup_steps``: while step < warmup, training is numerically identical
  to a no-gates run (gate=None threaded everywhere, no penalty, no gate updates)
  so variables become causally useful before pruning pressure applies;
- ``gate_lambda_ramp_steps``: after warmup, the effective lambda_gate scales
  linearly 0 -> config.lambda_gate;
- ``gate_clamp``: after each active optimizer step, log_alpha is clamped to
  [-clamp, +clamp] so neither saturation region kills the gradient.

Covers warmup equivalence, the ramp schedule, clamp bounds, and an end-to-end
demonstration that the schedule changes which basin the gates land in.
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
    gate_warmup_steps: int = 0,
    gate_lambda_ramp_steps: int = 0,
    gate_clamp: float | None = 3.0,
    use_gates: bool = True,
    k_max: int = 4,
    d: int = 8,
    seed: int = 0,
) -> JointTrainer:
    """A tiny gated (or ungated) JointTrainer on the XOR fake task."""
    torch.manual_seed(seed)
    task = XorTask(emb=2)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=d, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=k_max, v=2, n_labels=2)
    rot = OrthogonalRotation(d)
    layout = SubspaceLayout(d, k_max, init_width=1.0)
    gates = VariableGates(k_max, init=gate_init) if use_gates else None
    cfg = JointConfig(
        steps=steps,
        batch_size=32,
        n_sources=2,
        lr=1e-3,
        eval_every=0,
        seed=seed,
        device="cpu",
        use_gates=use_gates,
        lambda_gate=lambda_gate,
        gate_lr=gate_lr,
        gate_init=gate_init,
        gate_warmup_steps=gate_warmup_steps,
        gate_lambda_ramp_steps=gate_lambda_ramp_steps,
        gate_clamp=gate_clamp,
    )
    return JointTrainer(site, task, causal, rot, layout, cfg, gates=gates)


def _run_loop(trainer: JointTrainer) -> list[float]:
    """Run the manual loop mirroring JointTrainer.train; return per-step totals."""
    totals: list[float] = []
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch, step)
        trainer.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        trainer.optimizer.step()
        trainer._post_step(step)
        totals.append(float(losses["total"].item()))
    return totals


# ---------------------------------------------------------------------------
# warmup equivalence: a warmup-phase step is numerically identical to no-gates
# ---------------------------------------------------------------------------


def test_warmup_matches_no_gates_trajectory() -> None:
    """With gate_warmup_steps > steps, a gated trainer matches a no-gates run.

    Same seed / same modules init: per-step losses equal to float tolerance and
    log_alpha is exactly unchanged (gates receive no updates during warmup).
    """
    steps = 12
    gated = _build_trainer(
        steps=steps,
        lambda_gate=1.0,
        gate_lr=0.05,
        gate_warmup_steps=steps + 1,  # entire run is warmup
        gate_lambda_ramp_steps=0,
    )
    log_alpha_before = gated.gates.log_alpha.detach().clone()
    gated_totals = _run_loop(gated)

    nogate = _build_trainer(
        steps=steps,
        lambda_gate=1.0,
        gate_lr=0.05,
        use_gates=False,
    )
    nogate_totals = _run_loop(nogate)

    # log_alpha exactly unchanged: no gate updates happened during warmup.
    assert torch.equal(gated.gates.log_alpha.detach(), log_alpha_before)
    # Per-step losses identical to the no-gates run.
    assert len(gated_totals) == len(nogate_totals) == steps
    for i, (a, b) in enumerate(zip(gated_totals, nogate_totals)):
        assert abs(a - b) < 1e-6, f"step {i}: gated {a} vs no-gates {b}"


def test_warmup_then_active_starts_updating_gates() -> None:
    """After warmup ends, gate params begin to move (log_alpha changes)."""
    steps = 30
    warmup = 10
    trainer = _build_trainer(
        steps=steps,
        lambda_gate=1.0,
        gate_lr=0.05,
        gate_warmup_steps=warmup,
        gate_lambda_ramp_steps=0,
    )
    before = trainer.gates.log_alpha.detach().clone()
    _run_loop(trainer)
    after = trainer.gates.log_alpha.detach()
    assert not torch.equal(before, after), "gates should move once warmup ends"


# ---------------------------------------------------------------------------
# effective lambda schedule
# ---------------------------------------------------------------------------


def test_effective_lambda_schedule() -> None:
    """effective lambda is 0 in warmup, linear over the ramp, full after."""
    trainer = _build_trainer(
        steps=100,
        lambda_gate=0.2,
        gate_lr=0.05,
        gate_warmup_steps=10,
        gate_lambda_ramp_steps=20,
    )
    lam = 0.2
    # warmup: exactly 0.
    assert trainer._effective_lambda_gate(0) == 0.0
    assert trainer._effective_lambda_gate(9) == 0.0
    # ramp start (first active step): fraction 0 -> still ~0.
    assert abs(trainer._effective_lambda_gate(10) - 0.0 * lam) < 1e-9
    # ramp midpoint (10 steps into a 20-step ramp): half.
    assert abs(trainer._effective_lambda_gate(20) - 0.5 * lam) < 1e-9
    # ramp end and beyond: full lambda.
    assert abs(trainer._effective_lambda_gate(30) - lam) < 1e-9
    assert abs(trainer._effective_lambda_gate(99) - lam) < 1e-9


def test_effective_lambda_schedule_phases() -> None:
    """gate_phase reflects the warmup / ramp / active boundaries."""
    trainer = _build_trainer(
        steps=100,
        lambda_gate=0.2,
        gate_lr=0.05,
        gate_warmup_steps=10,
        gate_lambda_ramp_steps=20,
    )
    assert trainer._gate_phase(0) == "warmup"
    assert trainer._gate_phase(9) == "warmup"
    assert trainer._gate_phase(10) == "ramp"
    assert trainer._gate_phase(29) == "ramp"
    assert trainer._gate_phase(30) == "active"
    assert trainer._gate_phase(99) == "active"


def test_instant_lambda_no_schedule() -> None:
    """With warmup=0 and ramp=0 the effective lambda is full from step 0."""
    trainer = _build_trainer(
        steps=50,
        lambda_gate=0.3,
        gate_lr=0.05,
        gate_warmup_steps=0,
        gate_lambda_ramp_steps=0,
    )
    assert trainer._effective_lambda_gate(0) == 0.3
    assert trainer._gate_phase(0) == "active"


# ---------------------------------------------------------------------------
# log_alpha clamp
# ---------------------------------------------------------------------------


def test_clamp_bounds_log_alpha_under_huge_gradient() -> None:
    """A single huge-gradient step leaves log_alpha inside [-clamp, +clamp]."""
    clamp = 3.0
    trainer = _build_trainer(
        steps=1,
        lambda_gate=1.0,
        gate_lr=1.0,  # large lr so Adam moves ~1.0 per param this step
        gate_clamp=clamp,
    )
    # Inject an enormous gradient on log_alpha and step, then post-step clamp.
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.gates.log_alpha.grad = torch.full_like(
        trainer.gates.log_alpha, 1e6
    )
    trainer.optimizer.step()
    trainer._post_step(0)
    la = trainer.gates.log_alpha.detach()
    assert bool((la >= -clamp - 1e-6).all())
    assert bool((la <= clamp + 1e-6).all())
    # And a negative huge gradient pushes to the +clamp bound (grad descent).
    trainer.gates.log_alpha.grad = torch.full_like(
        trainer.gates.log_alpha, -1e6
    )
    trainer.optimizer.step()
    trainer._post_step(0)
    la = trainer.gates.log_alpha.detach()
    assert bool((la <= clamp + 1e-6).all())


def test_clamp_none_disables_clamping() -> None:
    """gate_clamp=None leaves log_alpha above where a clamp would pin it.

    Drive both a clamped and an unclamped trainer with the same huge negative
    gradient over several steps (each AdamW step moves log_alpha by ~lr toward
    +inf); the clamped run is pinned at +clamp, the unclamped run runs past it.
    """
    clamp = 3.0
    unclamped = _build_trainer(steps=5, lambda_gate=1.0, gate_lr=1.0, gate_clamp=None)
    clamped = _build_trainer(steps=5, lambda_gate=1.0, gate_lr=1.0, gate_clamp=clamp)
    for trainer in (unclamped, clamped):
        for _ in range(5):
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.gates.log_alpha.grad = torch.full_like(
                trainer.gates.log_alpha, -1e6
            )
            trainer.optimizer.step()
            trainer._post_step(0)
    # Clamped is pinned at +clamp; unclamped ran past it.
    assert bool((clamped.gates.log_alpha.detach() <= clamp + 1e-6).all())
    assert bool((unclamped.gates.log_alpha.detach() > clamp).all())


def test_no_clamp_during_warmup() -> None:
    """Warmup steps do not touch log_alpha even with a clamp set."""
    trainer = _build_trainer(
        steps=1,
        lambda_gate=1.0,
        gate_lr=1.0,
        gate_init=5.0,  # above the +3.0 clamp bound
        gate_clamp=3.0,
        gate_warmup_steps=10,
    )
    before = trainer.gates.log_alpha.detach().clone()
    trainer._post_step(0)  # step 0 is in warmup
    assert torch.equal(trainer.gates.log_alpha.detach(), before)


# ---------------------------------------------------------------------------
# end-to-end: the schedule changes the basin
# ---------------------------------------------------------------------------


def test_schedule_changes_basin_vs_instant_lambda() -> None:
    """Instant lambda collapses gates early; warmup+ramp keeps them live.

    A gate_init just above the live threshold plus an aggressive penalty and a
    fast gate_lr collapses all gates by the end of a short run. The same penalty
    behind a warmup (gates frozen, no penalty) followed by a ramp leaves the
    gates untouched through warmup so they remain live at the end of warmup, and
    the subsequent gentle ramp prunes only partially -- so the final gated_k is
    strictly larger than the instant-lambda run.
    """
    steps = 120
    k_max = 4
    warmup = 60
    # g_det(init) = clamp(sigmoid(init)*(zeta-gamma)+gamma, 0, 1); init=1.5 gives
    # g_det = sigmoid(1.5)*1.2 - 0.1 ~= 0.90 > 0.5, so all gates start live and a
    # warmup that freezes log_alpha keeps them all live.
    init = 1.5
    lam = 1.0
    glr = 0.05

    # Instant lambda: full penalty + fast lr from step 0 collapses every gate.
    instant = _build_trainer(
        steps=steps,
        lambda_gate=lam,
        gate_lr=glr,
        gate_init=init,
        gate_warmup_steps=0,
        gate_lambda_ramp_steps=0,
        gate_clamp=3.0,
        k_max=k_max,
        seed=0,
    )
    _run_loop(instant)
    gated_k_instant = instant.gates.gated_k()

    # Warmup+ramp: identical penalty/lr/init, but 60 warmup steps (gates frozen,
    # no penalty) then a 40-step ramp -- prunes only partially.
    scheduled = _build_trainer(
        steps=steps,
        lambda_gate=lam,
        gate_lr=glr,
        gate_init=init,
        gate_warmup_steps=warmup,
        gate_lambda_ramp_steps=40,
        gate_clamp=3.0,
        k_max=k_max,
        seed=0,
    )
    for step in range(warmup):
        scheduled._set_temperatures(step)
        batch = scheduled._sample_training_batch()
        losses = scheduled._compute_losses(batch, step)
        scheduled.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        scheduled.optimizer.step()
        scheduled._post_step(step)
    gated_k_after_warmup = scheduled.gates.gated_k()
    # Finish the run.
    for step in range(warmup, steps):
        scheduled._set_temperatures(step)
        batch = scheduled._sample_training_batch()
        losses = scheduled._compute_losses(batch, step)
        scheduled.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        scheduled.optimizer.step()
        scheduled._post_step(step)
    gated_k_scheduled = scheduled.gates.gated_k()

    # Warmup preserved liveness (log_alpha untouched -> still all live at init).
    assert gated_k_after_warmup == k_max, (
        f"warmup should preserve all {k_max} gates, got {gated_k_after_warmup}"
    )
    # Instant lambda collapses everything; the schedule prunes only partially.
    assert gated_k_instant == 0, (
        f"instant lambda should collapse all gates, got {gated_k_instant}"
    )
    assert 0 < gated_k_scheduled < k_max, (
        f"schedule should prune partially (0 < k < {k_max}), got {gated_k_scheduled}"
    )
    assert gated_k_instant < gated_k_scheduled, (
        f"instant lambda gated_k={gated_k_instant} should be < scheduled "
        f"gated_k={gated_k_scheduled}"
    )
