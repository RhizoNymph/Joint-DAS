"""Training smoke tests: loss decreases and Q receives gradients."""

from __future__ import annotations

import torch

from jdas.causal_model import LearnedCausalModel
from jdas.rotation import OrthogonalRotation, SubspaceLayout
from jdas.training import JointConfig, JointTrainer

from .fakes import MLPSite, XorTask


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
