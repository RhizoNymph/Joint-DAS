"""CPU-only tests for the price-tagging task (stub tokenizer)."""

from __future__ import annotations

import torch

from jdas.tasks.price_tagging import PriceTaggingTask
from jdas.types import InterventionBatch, Task


def test_protocol_conformance(tokenizer) -> None:
    task = PriceTaggingTask(tokenizer)
    assert isinstance(task, Task)
    assert task.n_labels == 2
    assert task.k_gt == 2


def test_label_matches_gt_variables(tokenizer) -> None:
    """gt_label_fn(gt_variables(inputs)) reproduces the sampled labels."""
    task = PriceTaggingTask(tokenizer)
    gen = torch.Generator().manual_seed(0)
    inputs, labels = task.sample_inputs(128, gen)
    vars_ = task.gt_variables(inputs)
    assert torch.equal(task.gt_label_fn(vars_), labels)


def test_and_semantics(tokenizer) -> None:
    """Label 1 == in range == (Z>=X) AND (Z<=Y)."""
    task = PriceTaggingTask(tokenizer)
    gen = torch.Generator().manual_seed(1)
    x, y, z = task._sample_xyz(256, gen, torch.device("cpu"))
    inputs = task._render_batch(x, y, z, torch.device("cpu"))
    labels = task._labels_from_xyz(x, y, z)
    expected = ((z >= x) & (z <= y)).long()
    assert torch.equal(labels, expected)
    # gt_variables recomputed from decoded features must agree too.
    vars_ = task.gt_variables(inputs)
    assert torch.equal(vars_[:, 0], (z >= x).long())
    assert torch.equal(vars_[:, 1], (z <= y).long())


def test_causal_features_roundtrip(tokenizer) -> None:
    """Decoding token ids recovers X, Y, Z (normalized by /10) to 2 decimals."""
    task = PriceTaggingTask(tokenizer)
    gen = torch.Generator().manual_seed(2)
    x, y, z = task._sample_xyz(64, gen, torch.device("cpu"))
    inputs = task._render_batch(x, y, z, torch.device("cpu"))
    feats = task.causal_features(inputs)  # (B, 3) normalized
    assert feats.shape == (64, 3)
    torch.testing.assert_close(feats[:, 0], x / 10.0, atol=1e-6, rtol=0)
    torch.testing.assert_close(feats[:, 1], y / 10.0, atol=1e-6, rtol=0)
    torch.testing.assert_close(feats[:, 2], z / 10.0, atol=1e-6, rtol=0)


def test_region_balance(tokenizer) -> None:
    """Z regions (below/inside/above) are ~1/3 each -> labels ~1/3 yes."""
    task = PriceTaggingTask(tokenizer)
    gen = torch.Generator().manual_seed(3)
    x, y, z = task._sample_xyz(6000, gen, torch.device("cpu"))
    below = (z < x).float().mean().item()
    inside = ((z >= x) & (z <= y)).float().mean().item()
    above = (z > y).float().mean().item()
    for frac in (below, inside, above):
        assert 0.28 < frac < 0.39, (below, inside, above)


def test_sample_batch_shapes(tokenizer) -> None:
    """Packed shapes: base (B,2,T), sources (B,m,2,T); assignment in range."""
    task = PriceTaggingTask(tokenizer)
    gen = torch.Generator().manual_seed(4)
    batch = task.sample_batch(8, n_sources=2, k_max=4, generator=gen)
    assert isinstance(batch, InterventionBatch)
    assert batch.base_inputs.dim() == 3 and batch.base_inputs.shape[0] == 8
    assert batch.base_inputs.shape[1] == 2
    assert batch.source_inputs.shape[0] == 8 and batch.source_inputs.shape[1] == 2
    assert batch.source_inputs.shape[2] == 2
    # base and sources share the same padded T.
    assert batch.base_inputs.shape[-1] == batch.source_inputs.shape[-1]
    assert batch.source_assignment.shape == (8, 4)
    assert int(batch.source_assignment.max()) < 2
    assert int(batch.source_assignment.min()) >= -1
    assert batch.base_labels.shape == (8,)
    assert batch.source_labels.shape == (8, 2)
    # labels consistent with gt.
    assert torch.equal(
        batch.base_labels, task.gt_label_fn(task.gt_variables(batch.base_inputs))
    )


# -- z_digits intervention position --------------------------------------------


def test_z_digits_produces_three_channels(tokenizer) -> None:
    """position='z_digits' packs a third (position) channel; 'last' stays 2-channel."""
    gen = torch.Generator().manual_seed(0)
    last_task = PriceTaggingTask(tokenizer, position="last")
    z_task = PriceTaggingTask(tokenizer, position="z_digits")
    x, y, z = last_task._sample_xyz(6, gen, torch.device("cpu"))
    last = last_task._render_batch(x, y, z, torch.device("cpu"))
    three = z_task._render_batch(x, y, z, torch.device("cpu"))
    assert last.shape[1] == 2
    assert three.shape[1] == 3
    # ids / mask channels are identical between the two renderings.
    assert torch.equal(three[:, 0], last[:, 0])
    assert torch.equal(three[:, 1], last[:, 1])


def test_z_digits_position_points_at_z_final_digit(tokenizer) -> None:
    """The packed position indexes the last token of the item amount Z, which
    decodes to a digit that appears in Z's rendered 2-decimal string."""
    task = PriceTaggingTask(tokenizer, position="z_digits")
    gen = torch.Generator().manual_seed(1)
    x, y, z = task._sample_xyz(16, gen, torch.device("cpu"))
    packed = task._render_batch(x, y, z, torch.device("cpu"))  # (B, 3, T)
    ids = packed[:, 0]
    pos = packed[:, 2, 0]  # per-example position
    for i in range(z.shape[0]):
        z_str = f"{float(z[i]):.2f}"
        tok_id = int(ids[i, int(pos[i])])
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        # For the char-level stub the final Z token is its last character (a
        # digit); assert it is a digit contained in the rendered Z string.
        assert decoded.isdigit(), (decoded, z_str)
        assert decoded in z_str, (decoded, z_str)
        # The final char of Z is the last decimal digit.
        assert decoded == z_str[-1]


def test_z_digits_sample_batch_shapes(tokenizer) -> None:
    """sample_batch with z_digits yields (B,3,T) base and (B,m,3,T) sources."""
    task = PriceTaggingTask(tokenizer, position="z_digits")
    gen = torch.Generator().manual_seed(2)
    batch = task.sample_batch(8, n_sources=2, k_max=4, generator=gen)
    assert batch.base_inputs.shape[1] == 3
    assert batch.source_inputs.shape[2] == 3
    # causal_features still decodes (X, Y, Z) from the 3-channel packing.
    feats = task.causal_features(batch.base_inputs)
    assert feats.shape == (8, 3)
