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
