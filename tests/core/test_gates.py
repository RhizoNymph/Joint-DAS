"""Tests for per-variable hard-concrete (L0) gates.

Covers:
- gate math (penalty monotone in log_alpha, sample in [0,1], exact 0/1
  reachable, gradient flows through both the sample and the penalty),
- N/H no-op symmetry (a hand-closed gate makes an interchange on that variable
  a no-op on BOTH N's masked activation and H's counterfactual label),
- live-restricted IIA (swap sampling restricted to live variables),
- checkpoint round-trip with gates and the gate/no-gate mismatch error,
- CLI wiring smoke test (argparse accepts --gates / --lambda-gate).
"""

from __future__ import annotations

import math

import pytest
import torch

from jdas.causal_model import LearnedCausalModel
from jdas.gates import VariableGates
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import (
    CheckpointError,
    JointConfig,
    JointTrainer,
    load_checkpoint,
    save_checkpoint,
)
from jdas.types import InterventionBatch

from .fakes import IdentitySite, MLPSite, XorTask


# ---------------------------------------------------------------------------
# gate math
# ---------------------------------------------------------------------------


def test_init_log_alpha_default() -> None:
    g = VariableGates(4)
    assert g.log_alpha.shape == (4,)
    assert torch.allclose(g.log_alpha, torch.full((4,), 2.0))


def test_constants_match_spec() -> None:
    g = VariableGates(2)
    assert g.beta == pytest.approx(2.0 / 3.0)
    assert g.gamma == pytest.approx(-0.1)
    assert g.zeta == pytest.approx(1.1)


def test_sample_in_unit_interval() -> None:
    """Stochastic samples are always clamped to [0, 1]."""
    torch.manual_seed(0)
    g = VariableGates(8)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        s = g.sample(generator=gen)
        assert s.shape == (8,)
        assert float(s.min()) >= 0.0
        assert float(s.max()) <= 1.0


def test_sample_reaches_exact_zero_and_one() -> None:
    """The (gamma, zeta) stretch lets the sample hit exactly 0 and 1."""
    torch.manual_seed(0)
    # Very open gate -> saturates to 1 often; very closed -> 0 often.
    g_open = VariableGates(1, init_log_alpha=20.0)
    g_closed = VariableGates(1, init_log_alpha=-20.0)
    gen = torch.Generator().manual_seed(1)
    saw_one = False
    saw_zero = False
    for _ in range(200):
        if float(g_open.sample(generator=gen)[0]) == 1.0:
            saw_one = True
        if float(g_closed.sample(generator=gen)[0]) == 0.0:
            saw_zero = True
    assert saw_one
    assert saw_zero


def test_deterministic_open_and_closed() -> None:
    """g_det > 0.5 (live) for large log_alpha, <= 0.5 (dead) for small."""
    g = VariableGates(2, init_log_alpha=0.0)
    with torch.no_grad():
        g.log_alpha.copy_(torch.tensor([10.0, -10.0]))
    det = g.deterministic()
    assert float(det[0]) > 0.5
    assert float(det[1]) <= 0.5
    live = g.live_mask()
    assert bool(live[0]) is True
    assert bool(live[1]) is False
    assert g.gated_k() == 1


def test_penalty_monotone_in_log_alpha() -> None:
    """L_gate is strictly increasing in each log_alpha (expected open count)."""
    lo = VariableGates(3, init_log_alpha=-3.0)
    hi = VariableGates(3, init_log_alpha=3.0)
    assert float(hi.penalty()) > float(lo.penalty())
    # And bounded by k on both sides of the sigmoid.
    assert 0.0 < float(lo.penalty()) < float(hi.penalty()) < 3.0


def test_penalty_gradient_flows_to_log_alpha() -> None:
    g = VariableGates(4)
    loss = g.penalty()
    loss.backward()
    assert g.log_alpha.grad is not None
    assert float(g.log_alpha.grad.abs().sum()) > 0.0


def test_sample_gradient_flows_to_log_alpha() -> None:
    """Gradient reaches log_alpha through the stochastic sample path."""
    g = VariableGates(4)
    gen = torch.Generator().manual_seed(3)
    s = g.sample(generator=gen)
    # A downstream scalar that depends on the (non-saturated) sample.
    s.sum().backward()
    assert g.log_alpha.grad is not None
    assert float(g.log_alpha.grad.abs().sum()) > 0.0


def test_hard_straight_through_value_and_grad() -> None:
    """hard(g) is 0/1 in the forward pass but passes g's gradient."""
    g = torch.tensor([0.2, 0.7, 0.9, 0.1], requires_grad=True)
    h = VariableGates.hard(g)
    assert torch.equal(h.detach(), torch.tensor([0.0, 1.0, 1.0, 0.0]))
    h.sum().backward()
    # d hard / d g == 1 (straight-through identity).
    assert torch.allclose(g.grad, torch.ones_like(g))


# ---------------------------------------------------------------------------
# N-side: gate-scaled effective widths
# ---------------------------------------------------------------------------


def test_layout_closed_gate_zeros_soft_mask() -> None:
    """A closed gate (g_i=0) collapses variable i's soft mask to ~0."""
    layout = SubspaceLayout(8, 3, init_width=2.0)
    gate = torch.tensor([1.0, 0.0, 1.0])
    masks = layout.soft_masks(gate=gate)
    assert masks.shape == (3, 8)
    assert float(masks[1].abs().max()) < 1e-4
    # Open variables keep nonzero mass.
    assert float(masks[0].abs().max()) > 0.5


def test_layout_closed_gate_zeros_hard_mask() -> None:
    layout = SubspaceLayout(8, 3, init_width=2.0)
    gate = torch.tensor([1.0, 0.0, 1.0])
    masks = layout.hard_masks(gate=gate)
    assert int(masks[1].sum()) == 0
    assert int(masks[0].sum()) >= 1


def test_layout_total_aligned_dims_respects_gate() -> None:
    layout = SubspaceLayout(12, 3, init_width=2.0)
    full = float(layout.total_aligned_dims())
    gated = float(layout.total_aligned_dims(gate=torch.tensor([1.0, 0.0, 0.0])))
    assert gated < full


# ---------------------------------------------------------------------------
# N/H no-op symmetry
# ---------------------------------------------------------------------------


def _identity_batch(bits: torch.Tensor, k_max: int, swap_var: int) -> InterventionBatch:
    """One example whose base/source differ, swapping only `swap_var`."""
    b = bits.shape[0]
    base = bits
    source = (1.0 - bits).unsqueeze(1)  # (b, 1, d) flipped source
    assign = torch.full((b, k_max), -1, dtype=torch.long)
    assign[:, swap_var] = 0
    return InterventionBatch(
        base_inputs=base,
        source_inputs=source,
        source_assignment=assign,
        base_labels=torch.zeros(b, dtype=torch.long),
        source_labels=torch.zeros(b, 1, dtype=torch.long),
    )


def test_n_side_closed_gate_is_noop() -> None:
    """Interchange on a gated-off variable does not change N's masked hidden."""
    from jdas.intervention import interchange

    d = 4
    site = IdentitySite(d=d, n_labels=2)
    with torch.no_grad():
        # Head reads every dim (incl. var1's dims 2,3) so a swap on var1 is
        # visible in the output when its gate is open.
        site.head.weight.copy_(
            torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
        )
    rot = OrthogonalRotation(d)
    rot.set_matrix(torch.eye(d))
    layout = SubspaceLayout(d, 2, init_width=2.0)  # var0 -> dims [0,2), var1 -> [2,4)

    base = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    batch = _identity_batch(base, k_max=2, swap_var=1)

    # Gate variable 1 CLOSED: swapping it must be a no-op on N.
    gate_closed = torch.tensor([1.0, 0.0])
    out_closed = interchange(site, rot, layout, batch, hard=True, gate=gate_closed)
    # Baseline: no swap at all.
    no_swap = _identity_batch(base, k_max=2, swap_var=1)
    no_swap.source_assignment[:] = -1
    out_none = interchange(site, rot, layout, no_swap, hard=True, gate=gate_closed)
    torch.testing.assert_close(out_closed, out_none)

    # Sanity: with the gate OPEN the same swap DOES change the output.
    gate_open = torch.tensor([1.0, 1.0])
    out_open = interchange(site, rot, layout, batch, hard=True, gate=gate_open)
    assert not torch.allclose(out_open, out_none)


def test_h_side_closed_gate_is_noop() -> None:
    """Counterfactual label is unchanged when a swapped variable is gated off."""
    torch.manual_seed(0)
    model = LearnedCausalModel(input_dim=6, k_max=3, v=2, n_labels=2)
    base = torch.randn(5, 6)
    source = torch.randn(5, 2, 6)
    # Swap only variable 1.
    assign = torch.full((5, 3), -1, dtype=torch.long)
    assign[:, 1] = 0
    gate_closed = torch.tensor([1.0, 0.0, 1.0])

    cf = model.counterfactual_predict(base, source, assign, gate=gate_closed).argmax(-1)
    # No-swap baseline under the same gate.
    no_assign = torch.full((5, 3), -1, dtype=torch.long)
    base_pred = model.counterfactual_predict(base, source, no_assign, gate=gate_closed)
    assert torch.equal(cf, base_pred.argmax(-1))


def test_h_side_dead_variable_is_constant_zero() -> None:
    """A gated-off variable contributes its 0-value one-hot regardless of input."""
    torch.manual_seed(1)
    model = LearnedCausalModel(input_dim=6, k_max=2, v=2, n_labels=2)
    x = torch.randn(4, 6)
    gate = torch.tensor([1.0, 0.0])
    # Manually build the masked one-hots and compare to model.predict(gate=...).
    onehots = model.variables(x)
    masked = onehots * VariableGates.hard(gate).view(1, -1, 1)
    expected = model.decode(masked)
    got = model.predict(x, gate=gate)
    torch.testing.assert_close(got, expected)
    # variable 1's contribution is the all-zero one-hot (constant across inputs).
    assert torch.allclose(masked[:, 1, :], torch.zeros(4, 2))


# ---------------------------------------------------------------------------
# live-restricted IIA
# ---------------------------------------------------------------------------


def test_live_iia_restricts_to_live_variables() -> None:
    """iia_live only swaps live variables; a dead var never appears in a swap."""
    from jdas.eval import iia_live

    task = XorTask(emb=1)
    d = 2

    from torch import nn

    class RiggedSite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d = d
            self.n_labels = 2

        def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.reshape(inputs.shape[0], 2)

        def logits_with_hidden(self, inputs, hidden):
            y = (hidden[:, 0] > 0).long() ^ (hidden[:, 1] > 0).long()
            return torch.nn.functional.one_hot(y, 2).float() * 10.0

        def logits(self, inputs):
            return self.logits_with_hidden(inputs, self.hidden(inputs))

    from jdas.causal_model import FixedCausalModel

    def _bits(inputs):
        x = inputs.reshape(inputs.shape[0], 2, 1)
        return (x.mean(dim=-1) > 0).long()

    model = FixedCausalModel(_bits, task.label_from_variables, k=2, v=2, n_labels=2)
    site = RiggedSite()
    rot = OrthogonalRotation(d)
    rot.set_matrix(torch.eye(d))
    layout = SubspaceLayout(d, 2, init_width=1.0)
    with torch.no_grad():
        raw = torch.log(torch.expm1(torch.tensor(1.0)))
        layout.raw_widths.fill_(float(raw))

    # Only variable 0 is live.
    gates = VariableGates(2, init_log_alpha=0.0)
    with torch.no_grad():
        gates.log_alpha.copy_(torch.tensor([10.0, -10.0]))

    gen = torch.Generator().manual_seed(4)
    scores = iia_live(
        site, rot, layout, model, task, gates,
        n_batches=4, batch_size=64, n_sources=2, generator=gen, swap_sizes=(1, 2),
    )
    # Only one live variable -> |I|=2 is not applicable.
    assert 1 in scores
    assert scores.get(2) is None
    assert scores[1] > 0.95


# ---------------------------------------------------------------------------
# trainer wiring + checkpointing
# ---------------------------------------------------------------------------


def _gated_trainer(steps: int, lambda_gate: float) -> JointTrainer:
    torch.manual_seed(0)
    task = XorTask(emb=3)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=8, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=4, v=2, n_labels=2)
    rot = OrthogonalRotation(8)
    layout = SubspaceLayout(8, 4, init_width=1.0)
    gates = VariableGates(4)
    cfg = JointConfig(
        steps=steps, batch_size=32, n_sources=2, lr=5e-3, eval_every=0,
        seed=0, device="cpu", use_gates=True, lambda_gate=lambda_gate,
    )
    return JointTrainer(site, task, causal, rot, layout, cfg, gates=gates)


def test_trainer_requires_gates_when_use_gates() -> None:
    """use_gates=True without a gates module is a configuration error."""
    from jdas.training import TrainingError

    torch.manual_seed(0)
    task = XorTask(emb=3)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=8, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=4, v=2, n_labels=2)
    rot = OrthogonalRotation(8)
    layout = SubspaceLayout(8, 4, init_width=1.0)
    cfg = JointConfig(steps=1, use_gates=True, lambda_gate=0.1)
    with pytest.raises(TrainingError):
        JointTrainer(site, task, causal, rot, layout, cfg)


def test_trainer_gate_penalty_in_loss_and_grad() -> None:
    """With gates on, the gate penalty enters the loss and log_alpha gets grad."""
    trainer = _gated_trainer(steps=1, lambda_gate=0.1)
    trainer._set_temperatures(0)
    batch = trainer._sample_training_batch()
    losses = trainer._compute_losses(batch)
    assert "l_gate" in losses
    assert float(losses["l_gate"].item()) > 0.0
    trainer.optimizer.zero_grad()
    losses["total"].backward()
    assert trainer.gates.log_alpha.grad is not None
    assert float(trainer.gates.log_alpha.grad.abs().sum()) > 0.0


def test_trainer_gates_move_with_penalty() -> None:
    """A nonzero lambda_gate pushes log_alpha down (some gates start closing)."""
    trainer = _gated_trainer(steps=40, lambda_gate=0.5)
    before = trainer.gates.log_alpha.detach().clone()
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch)
        trainer.optimizer.zero_grad()
        losses["total"].backward()
        trainer.optimizer.step()
    after = trainer.gates.log_alpha.detach()
    assert float(after.mean()) < float(before.mean())


def test_history_records_gate_stats() -> None:
    """train() history/final carry gated_k and g_det when gates are on."""
    trainer = _gated_trainer(steps=6, lambda_gate=0.1)
    trainer.config.eval_every = 3
    out = trainer.train()
    assert "gated_k" in out["final"]
    assert "g_det" in out["final"]
    assert len(out["final"]["g_det"]) == 4
    assert "iia_1_live" in out["final"]


def test_checkpoint_roundtrip_with_gates(tmp_path) -> None:
    """save/load restores gate log_alpha; live mask is preserved."""
    torch.manual_seed(0)
    d = 8
    causal = LearnedCausalModel(input_dim=6, k_max=4, v=2, n_labels=2, decoder_hidden=None)
    rot = OrthogonalRotation(d)
    layout = SubspaceLayout(d, 4, init_width=1.5)
    gates = VariableGates(4)
    with torch.no_grad():
        gates.log_alpha.copy_(torch.tensor([5.0, -5.0, 3.0, -3.0]))
    cfg = JointConfig(steps=1, use_gates=True, lambda_gate=0.1)

    path = tmp_path / "ckpt_gate.pt"
    save_checkpoint(path, rot, layout, causal, cfg, gates=gates)
    loaded = load_checkpoint(path)
    g2 = loaded["gates"]
    assert g2 is not None
    torch.testing.assert_close(g2.log_alpha, gates.log_alpha, atol=1e-6, rtol=0)
    assert torch.equal(g2.live_mask(), gates.live_mask())


def test_checkpoint_mismatch_gate_into_nogate(tmp_path) -> None:
    """Loading a gated checkpoint but omitting gates (or vice versa) raises."""
    torch.manual_seed(0)
    causal = LearnedCausalModel(input_dim=6, k_max=3, v=2, n_labels=2, decoder_hidden=None)
    rot = OrthogonalRotation(8)
    layout = SubspaceLayout(8, 3, init_width=1.5)
    cfg = JointConfig(steps=1)

    # Saved WITHOUT gates.
    path = tmp_path / "ckpt_nogate.pt"
    save_checkpoint(path, rot, layout, causal, cfg)
    # Requesting gates from a gate-less checkpoint must error.
    with pytest.raises(CheckpointError):
        load_checkpoint(path, expect_gates=True)

    # Saved WITH gates.
    gates = VariableGates(3)
    path2 = tmp_path / "ckpt_gate2.pt"
    save_checkpoint(path2, rot, layout, causal, cfg, gates=gates)
    with pytest.raises(CheckpointError):
        load_checkpoint(path2, expect_gates=False)


# ---------------------------------------------------------------------------
# CLI wiring smoke
# ---------------------------------------------------------------------------


def test_run_toy_cli_accepts_gate_flags() -> None:
    """run_toy's parser accepts --gates and --lambda-gate."""
    from jdas.cli import runners

    parser = runners.build_toy_parser()
    args = parser.parse_args(
        ["--task", "hierarchical_equality", "--method", "joint",
         "--gates", "--lambda-gate", "0.1"]
    )
    assert args.gates is True
    assert args.lambda_gate == pytest.approx(0.1)


def test_run_toy_gates_rejected_for_fixed_h() -> None:
    """--gates with a fixed-H method is rejected."""
    from jdas.cli import runners

    with pytest.raises(SystemExit):
        runners._validate_gate_method("das_true", gates=True)
