"""Training smoke tests: loss decreases and Q receives gradients."""

from __future__ import annotations

import pytest
import torch

from jdas.causal_model import FixedCausalModel, LearnedCausalModel
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import (
    DASTrainer,
    JointConfig,
    JointTrainer,
    load_checkpoint,
    save_checkpoint,
)

from .fakes import IdentitySite, MLPSite, XorTask


def _build(steps: int) -> JointTrainer:
    torch.manual_seed(0)
    task = XorTask(emb=3)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=8, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=3, v=2, n_labels=2)
    rot = OrthogonalRotation(8)
    layout = SubspaceLayout(8, 3, init_width=1.5)
    cfg = JointConfig(
        steps=steps,
        batch_size=32,
        n_sources=2,
        lr=5e-3,
        eval_every=0,  # skip eval during the loop for speed
        seed=0,
        device="cpu",
    )
    return JointTrainer(site, task, causal, rot, layout, cfg)


def test_joint_step_gives_rotation_gradients() -> None:
    """A single joint step produces nonzero gradients on Q."""
    trainer = _build(steps=1)
    trainer._set_temperatures(0)
    batch = trainer._sample_training_batch()
    losses = trainer._compute_losses(batch)
    trainer.optimizer.zero_grad()
    losses["total"].backward()
    rot_grads = [p.grad for p in trainer.rotation.parameters() if p.grad is not None]
    assert rot_grads and any(g.abs().sum() > 0 for g in rot_grads)
    # Causal-model params also get gradients.
    cm_grads = [p.grad for p in trainer.causal_model.parameters() if p.grad is not None]
    assert cm_grads and any(g.abs().sum() > 0 for g in cm_grads)


def test_joint_loss_decreases() -> None:
    """Over a handful of steps the total loss trends down."""
    trainer = _build(steps=60)
    first_losses = []
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch)
        trainer.optimizer.zero_grad()
        losses["total"].backward()
        trainer.optimizer.step()
        first_losses.append(float(losses["total"].item()))
    early = sum(first_losses[:10]) / 10
    late = sum(first_losses[-10:]) / 10
    assert late < early


def test_joint_train_returns_serializable_history() -> None:
    """trainer.train() returns a JSON-serializable history + final metrics."""
    import json

    torch.manual_seed(0)
    task = XorTask(emb=3)
    input_dim = 2 * task.emb
    site = MLPSite(in_dim=input_dim, d=8, n_labels=2, seed=0)
    causal = LearnedCausalModel(input_dim=input_dim, k_max=3, v=2, n_labels=2)
    rot = OrthogonalRotation(8)
    layout = SubspaceLayout(8, 3, init_width=1.5)
    cfg = JointConfig(steps=12, batch_size=16, n_sources=2, eval_every=6, seed=0)
    trainer = JointTrainer(site, task, causal, rot, layout, cfg)
    out = trainer.train()
    assert "history" in out and "final" in out
    json.dumps(out)  # must not raise
    assert 0.0 <= out["final"]["iia_1"] <= 1.0


def test_rotation_stays_orthogonal_during_training() -> None:
    """Q remains orthogonal after real optimizer steps in the trainer."""
    trainer = _build(steps=20)
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch)
        trainer.optimizer.zero_grad()
        losses["total"].backward()
        trainer.optimizer.step()
    q = trainer.rotation.matrix
    eye = torch.eye(q.shape[0])
    assert torch.allclose(q @ q.T, eye, atol=1e-4)


# -- sparse_mode ---------------------------------------------------------------


def _flat_cf_trainer(sparse_mode: str, lambda_sparse: float, steps: int) -> DASTrainer:
    """DAS trainer whose cf-loss is flat w.r.t. the layout.

    The site's head weight is zeroed so its logits are constant regardless of the
    injected hidden -> L_cf carries no gradient to the layout, isolating the
    sparsity penalty.  The fixed model emits a constant label.  Only the layout
    trains (rotation frozen, model fixed).
    """
    torch.manual_seed(0)
    task = XorTask(emb=3)
    # IdentitySite.hidden returns the flattened input; XorTask inputs are
    # (B, 2, 3) = 6 dims, so the site dim must be 6.
    d = 2 * task.emb
    site = IdentitySite(d=d, n_labels=2)
    with torch.no_grad():
        site.head.weight.zero_()  # logits always 0 -> cf-loss flat

    def const_vars(inputs: torch.Tensor) -> torch.Tensor:
        b = inputs.shape[0]
        return torch.zeros(b, 2, dtype=torch.long)

    def const_label(vals: torch.Tensor) -> torch.Tensor:
        return torch.zeros(vals.shape[0], dtype=torch.long)

    model = FixedCausalModel(const_vars, const_label, k=2, v=2, n_labels=2)
    rot = OrthogonalRotation(d, freeze=True)
    layout = SubspaceLayout(d, 2, init_width=2.0)
    cfg = JointConfig(
        steps=steps,
        batch_size=16,
        n_sources=2,
        lr=0.2,
        lambda_sparse=lambda_sparse,
        sparse_mode=sparse_mode,
        eval_every=0,
        seed=0,
    )
    return DASTrainer(site, task, model, rot, layout, cfg)


def test_per_dim_sparse_shrinks_widths() -> None:
    """With per_dim mode and meaningful lambda, widths shrink when cf-loss is flat."""
    trainer = _flat_cf_trainer("per_dim", lambda_sparse=0.3, steps=80)
    init_total = float(trainer.layout.total_aligned_dims().item())
    for step in range(trainer.config.steps):
        trainer._set_temperatures(step)
        batch = trainer._sample_training_batch()
        losses = trainer._compute_losses(batch)
        trainer.optimizer.zero_grad()
        losses["total"].backward()
        trainer.optimizer.step()
    final_total = float(trainer.layout.total_aligned_dims().item())
    assert final_total < init_total - 1.0, (init_total, final_total)


def test_normalized_matches_old_sparse_value() -> None:
    """normalized mode reproduces the old L_sparse == total_aligned_dims / d."""
    trainer = _flat_cf_trainer("normalized", lambda_sparse=0.1, steps=1)
    batch = trainer._sample_training_batch()
    losses = trainer._compute_losses(batch)
    expected = float(
        trainer.layout.total_aligned_dims().item() / trainer.layout.d
    )
    assert float(losses["l_sparse"].item()) == pytest.approx(expected, rel=1e-5)


def test_per_dim_sparse_value_unnormalized() -> None:
    """per_dim L_sparse equals total_aligned_dims (no division by d)."""
    trainer = _flat_cf_trainer("per_dim", lambda_sparse=0.1, steps=1)
    batch = trainer._sample_training_batch()
    losses = trainer._compute_losses(batch)
    expected = float(trainer.layout.total_aligned_dims().item())
    assert float(losses["l_sparse"].item()) == expected


# -- checkpointing -------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path) -> None:
    """save/load reproduces hard widths, Q, and causal-model predictions."""
    torch.manual_seed(0)
    d = 8
    task = XorTask(emb=3)
    input_dim = 2 * task.emb
    causal = LearnedCausalModel(
        input_dim=input_dim, k_max=3, v=2, n_labels=2, decoder_hidden=None
    )
    rot = OrthogonalRotation(d)
    layout = SubspaceLayout(d, 3, init_width=2.0, max_width=6.0)
    cfg = JointConfig(steps=1, sparse_mode="per_dim", seed=0)

    # Fixed reference batch for causal-model predictions.
    gen = torch.Generator().manual_seed(7)
    ref = task.sample_batch(5, 2, 3, gen)
    pred_before = causal.predict(ref.base_inputs)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, rot, layout, causal, cfg, extra={"tag": "t"})

    loaded = load_checkpoint(path)
    r2, l2, c2 = loaded["rotation"], loaded["layout"], loaded["causal_model"]

    assert l2.max_width == 6.0
    assert torch.equal(l2.hard_widths(), layout.hard_widths())
    torch.testing.assert_close(r2.matrix, rot.matrix, atol=1e-6, rtol=0)
    torch.testing.assert_close(c2.predict(ref.base_inputs), pred_before, atol=1e-6, rtol=0)
    assert loaded["config"]["sparse_mode"] == "per_dim"
    assert loaded["meta"]["extra"]["tag"] == "t"


def test_checkpoint_roundtrip_mlp_decoder(tmp_path) -> None:
    """Round-trip also works with the default MLP decoder (decoder_hidden set)."""
    torch.manual_seed(1)
    causal = LearnedCausalModel(
        input_dim=6, k_max=2, v=2, n_labels=2, encoder_hidden=16, decoder_hidden=8
    )
    rot = OrthogonalRotation(6)
    layout = SubspaceLayout(6, 2, init_width=1.5)
    cfg = JointConfig(steps=1)
    x = torch.randn(4, 6)
    pred_before = causal.predict(x)

    path = tmp_path / "ckpt2.pt"
    save_checkpoint(path, rot, layout, causal, cfg)
    loaded = load_checkpoint(path)
    torch.testing.assert_close(
        loaded["causal_model"].predict(x), pred_before, atol=1e-6, rtol=0
    )
    assert loaded["causal_model"].decoder[0].out_features == 8
