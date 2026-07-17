"""CPU-only tests for HFSite on a tiny randomly-initialized Qwen2 model."""

from __future__ import annotations

import torch

from jdas.models.hf import HFSite
from jdas.tasks.price_tagging import PriceTaggingTask
from jdas.types import InterventionSite


def _make(tokenizer, tiny_model, layer: int = 2) -> tuple[HFSite, PriceTaggingTask]:
    site = HFSite(tiny_model, tokenizer, layer=layer, device="cpu")
    task = PriceTaggingTask(tokenizer)
    return site, task


def _inputs(task: PriceTaggingTask, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x, y, z = task._sample_xyz(n, gen, torch.device("cpu"))
    return task._render_batch(x, y, z, torch.device("cpu"))


def test_protocol_and_dims(tokenizer, tiny_model) -> None:
    site, _ = _make(tokenizer, tiny_model)
    assert isinstance(site, InterventionSite)
    assert site.d == tiny_model.config.hidden_size
    assert site.n_labels == 2
    assert len(site.yes_ids) >= 1 and len(site.no_ids) >= 1


def test_logits_with_own_hidden_matches_clean(tokenizer, tiny_model) -> None:
    """Injecting hidden(inputs) reproduces the clean logits (atol 1e-4)."""
    site, task = _make(tokenizer, tiny_model)
    inputs = _inputs(task, 6, seed=10)
    clean = site.logits(inputs)
    h = site.hidden(inputs)
    reinj = site.logits_with_hidden(inputs, h)
    torch.testing.assert_close(reinj, clean, atol=1e-4, rtol=1e-4)


def test_different_hidden_changes_logits(tokenizer, tiny_model) -> None:
    """Substituting another input's hidden changes the logits."""
    site, task = _make(tokenizer, tiny_model)
    a = _inputs(task, 6, seed=11)
    b = _inputs(task, 6, seed=12)
    # tokenize a and b to same length by re-rendering jointly is unnecessary:
    # forward uses each input's own ids; just ensure same T by padding here.
    # Pad both to the same T so we can inject b's hidden into a's forward.
    t = max(a.shape[-1], b.shape[-1])
    a = _pad_to(a, t)
    b = _pad_to(b, t)
    logits_a = site.logits(a)
    h_b = site.hidden(b)
    swapped = site.logits_with_hidden(a, h_b)
    assert not torch.allclose(swapped, logits_a, atol=1e-3)


def test_gradients_flow_to_injected_hidden(tokenizer, tiny_model) -> None:
    """Gradient reaches an injected hidden; model weights stay grad-free."""
    site, task = _make(tokenizer, tiny_model)
    inputs = _inputs(task, 4, seed=13)
    h = site.hidden(inputs).clone().requires_grad_(True)
    logits = site.logits_with_hidden(inputs, h)
    loss = logits[:, 1].sum()
    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert float(h.grad.abs().sum()) > 0.0
    # frozen model weights accumulate no gradient.
    for p in site.model.parameters():
        assert p.grad is None


def test_last_position_capture_under_left_padding(tokenizer, tiny_model) -> None:
    """Two prompts of different length in one batch each capture their own
    final prompt token (left padding puts it at position -1)."""
    site, task = _make(tokenizer, tiny_model)
    # Two prompts with clearly different token lengths.
    short = task.render_prompt(1.00, 2.00, 1.50)
    long = task.render_prompt(1.00, 2.00, 1.50) + " extra text here to lengthen it"
    packed = task._tokenize([short, long], torch.device("cpu"))  # (2, 2, T)
    assert packed.shape[0] == 2

    # Per-example single forwards (no padding) give the reference last-token
    # hiddens.  Compare to the batched (left-padded) capture.
    single_short = task._tokenize([short], torch.device("cpu"))
    single_long = task._tokenize([long], torch.device("cpu"))
    ref_short = site.hidden(single_short)[0]
    ref_long = site.hidden(single_long)[0]

    batched = site.hidden(packed)  # (2, d)
    torch.testing.assert_close(batched[0], ref_short, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(batched[1], ref_long, atol=1e-4, rtol=1e-4)


def _pad_to(packed: torch.Tensor, t: int) -> torch.Tensor:
    """Left-pad a packed (B, 2, T0) tensor to length t (pad id 0, mask 0)."""
    b, _, t0 = packed.shape
    if t0 == t:
        return packed
    out = torch.zeros(b, 2, t, dtype=packed.dtype)
    out[:, :, t - t0:] = packed
    return out


def _add_position_channel(packed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Append a broadcast per-example position channel -> (B, 3, T)."""
    b, _, t = packed.shape
    pos_channel = positions.view(b, 1).expand(b, t)
    return torch.cat([packed, pos_channel.unsqueeze(1).to(packed.dtype)], dim=1)


# -- per-example intervention position (B, 3, T) --------------------------------


def test_position_channel_last_matches_two_channel(tokenizer, tiny_model) -> None:
    """With positions == last index, (B,3,T) path equals the old (B,2,T) path."""
    site, task = _make(tokenizer, tiny_model)
    inputs = _inputs(task, 5, seed=20)  # (B, 2, T)
    t = inputs.shape[-1]
    positions = torch.full((inputs.shape[0],), t - 1, dtype=torch.long)
    three = _add_position_channel(inputs, positions)

    torch.testing.assert_close(site.hidden(three), site.hidden(inputs), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(site.logits(three), site.logits(inputs), atol=1e-5, rtol=1e-5)
    h = site.hidden(inputs)
    torch.testing.assert_close(
        site.logits_with_hidden(three, h),
        site.logits_with_hidden(inputs, h),
        atol=1e-5,
        rtol=1e-5,
    )


def test_reinject_own_hidden_at_arbitrary_positions(tokenizer, tiny_model) -> None:
    """logits_with_hidden(inputs, hidden(inputs)) ~= logits(inputs) at mixed
    per-example positions with a left-padded batch."""
    site, task = _make(tokenizer, tiny_model)
    # Mixed-length prompts -> left padding puts real tokens at different offsets.
    short = task.render_prompt(1.00, 2.00, 1.50)
    long = task.render_prompt(1.00, 2.00, 1.50) + " extra text to make it longer here"
    packed = task._tokenize([short, long], torch.device("cpu"))  # (2, 2, T)
    t = packed.shape[-1]
    # Pick a valid (non-pad) position per example: somewhere inside the prompt.
    mask = packed[:, 1]  # (2, T)
    # first non-pad column per row:
    first = torch.stack([m.nonzero()[0, 0] for m in mask])
    positions = ((first + t - 1) // 2).long()  # mid-ish, guaranteed valid
    three = _add_position_channel(packed, positions)

    clean = site.logits(three)
    reinj = site.logits_with_hidden(three, site.hidden(three))
    torch.testing.assert_close(reinj, clean, atol=1e-4, rtol=1e-4)


def test_injection_position_matters(tokenizer, tiny_model) -> None:
    """Injecting a vector at position p changes logits, and the right position's
    result differs from the wrong position's result."""
    site, task = _make(tokenizer, tiny_model)
    inputs = _inputs(task, 4, seed=21)
    t = inputs.shape[-1]
    b = inputs.shape[0]

    p_right = torch.full((b,), t - 1, dtype=torch.long)
    p_wrong = torch.full((b,), t - 3, dtype=torch.long)
    three_right = _add_position_channel(inputs, p_right)
    three_wrong = _add_position_channel(inputs, p_wrong)

    base = site.logits(inputs)
    inject = torch.randn(b, site.d)
    out_right = site.logits_with_hidden(three_right, inject)
    out_wrong = site.logits_with_hidden(three_wrong, inject)
    # Injecting a different vector changes the logits.
    assert not torch.allclose(out_right, base, atol=1e-3)
    # Injecting at the wrong position is not the same as the right position.
    assert not torch.allclose(out_right, out_wrong, atol=1e-3)


def test_gradients_flow_at_per_example_position(tokenizer, tiny_model) -> None:
    """Gradients reach a hidden injected at a per-example position."""
    site, task = _make(tokenizer, tiny_model)
    inputs = _inputs(task, 4, seed=22)
    t = inputs.shape[-1]
    b = inputs.shape[0]
    positions = torch.tensor([t - 1, t - 2, t - 1, t - 3])[:b]
    three = _add_position_channel(inputs, positions)
    h = site.hidden(three).clone().requires_grad_(True)
    logits = site.logits_with_hidden(three, h)
    logits[:, 1].sum().backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert float(h.grad.abs().sum()) > 0.0
    for p in site.model.parameters():
        assert p.grad is None
