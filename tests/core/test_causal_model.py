"""Straight-through causal-model counterfactual semantics on hand-built tables."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jdas.causal_model import FixedCausalModel, LearnedCausalModel


def _xor_label(vars: torch.Tensor) -> torch.Tensor:
    return (vars[:, 0] ^ vars[:, 1]).long()


def _bits_from_input(inputs: torch.Tensor) -> torch.Tensor:
    """Interpret a (B, 2) input as two bits (>0 -> 1)."""
    return (inputs > 0).long()


def test_fixed_model_predict_xor() -> None:
    """FixedCausalModel decodes XOR truth table correctly."""
    model = FixedCausalModel(_bits_from_input, _xor_label, k=2, v=2, n_labels=2)
    inputs = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    preds = model.predict(inputs).argmax(-1)
    assert preds.tolist() == [0, 1, 1, 0]


def test_fixed_model_counterfactual_xor() -> None:
    """Swapping var0 from a source flips the XOR output per truth table.

    base = (0,0) -> y=0.  source = (1, .) so z0(source)=1.  Swap var0 -> mixed
    (1,0) -> y=1.
    """
    model = FixedCausalModel(_bits_from_input, _xor_label, k=2, v=2, n_labels=2)
    base = torch.tensor([[-1.0, -1.0]])  # bits (0,0)
    source = torch.tensor([[[1.0, -1.0]]])  # (1,1,2) bits (1,0)
    assign = torch.tensor([[0, -1]])  # swap var0 from src0
    out = model.counterfactual_predict(base, source, assign)
    assert out.argmax(-1).item() == 1  # (1,0) XOR = 1


def test_fixed_model_counterfactual_both_vars_two_sources() -> None:
    """Swap both vars from two distinct sources -> uses their bits."""
    model = FixedCausalModel(_bits_from_input, _xor_label, k=2, v=2, n_labels=2)
    base = torch.tensor([[-1.0, -1.0]])  # (0,0)
    # src0 bits (1,?) -> var0=1; src1 bits (?,1) -> var1=1
    source = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])  # (1,2,2)
    assign = torch.tensor([[0, 1]])
    out = model.counterfactual_predict(base, source, assign)
    assert out.argmax(-1).item() == 0  # (1,1) XOR = 0


def test_learned_model_straight_through_forward_hard() -> None:
    """variables() are exact one-hots in the forward pass."""
    torch.manual_seed(0)
    model = LearnedCausalModel(input_dim=4, k_max=3, v=2, n_labels=2)
    inputs = torch.randn(5, 4)
    z = model.variables(inputs)
    assert z.shape == (5, 3, 2)
    # Each variable is a one-hot (sums to 1, entries in {0,1}).
    assert torch.allclose(z.sum(-1), torch.ones(5, 3))
    assert torch.all((z == 0) | (z == 1))


def test_learned_model_gradients_flow_through_st() -> None:
    """Gradients reach encoder + decoder params through straight-through."""
    torch.manual_seed(1)
    model = LearnedCausalModel(input_dim=4, k_max=2, v=2, n_labels=2)
    inputs = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    loss = F.cross_entropy(model.predict(inputs), labels)
    loss.backward()
    enc_grad = model.encoders[0][0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0


def test_learned_model_counterfactual_matches_manual_swap() -> None:
    """counterfactual_predict equals decode() of a manually swapped one-hot."""
    torch.manual_seed(2)
    model = LearnedCausalModel(input_dim=4, k_max=2, v=2, n_labels=2)
    base = torch.randn(3, 4)
    source = torch.randn(3, 2, 4)
    assign = torch.tensor([[1, -1], [-1, 0], [0, 1]])

    with torch.no_grad():
        base_z = model.variables(base)
        src_z = model.variables(source.reshape(6, 4)).reshape(3, 2, 2, 2)
        expected_z = base_z.clone()
        for b in range(3):
            for i in range(2):
                j = assign[b, i].item()
                if j >= 0:
                    expected_z[b, i] = src_z[b, j, i]
        expected = model.decode(expected_z)
        got = model.counterfactual_predict(base, source, assign)
    assert torch.allclose(got, expected, atol=1e-5)


def test_learned_model_decoder_can_realize_xor() -> None:
    """A hand-set decoder over one-hots realizes XOR; counterfactuals follow it."""
    model = LearnedCausalModel(input_dim=4, k_max=2, v=2, n_labels=2, decoder_hidden=None)
    # decoder input layout: [z0=0, z0=1, z1=0, z1=1] -> logits[2]
    # want argmax = z0 XOR z1.  Set weights so logit1 - logit0 = large when xor.
    with torch.no_grad():
        w = torch.zeros(2, 4)
        # logit for class 1 high when (z0=1,z1=0) or (z0=0,z1=1)
        # Use w[class1] . onehot; encode via: class1 = z0!=z1.
        # class1 score = a*(z0=1)+a*(z1=1) ; class0 score = a*(z0=1&z1=1)... simpler:
        # score1 = 1*(z0=1) + 1*(z1=1) - 2*? cannot cross terms w/ linear+onehot per-var.
        # Instead: class1 - class0 = z0_1 + z1_1 - 2*z0_1*z1_1 not linear. So use decoder_hidden.
        pass
    # Linear over per-variable one-hots cannot express XOR; verify MLP decoder can.
    model = LearnedCausalModel(input_dim=4, k_max=2, v=2, n_labels=2, decoder_hidden=16)
    # Train the decoder briefly to fit XOR of fixed variable one-hots.
    opt = torch.optim.Adam(model.decoder.parameters(), lr=0.05)
    table = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    onehots = F.one_hot(table, num_classes=2).float()  # (4,2,2)
    ys = torch.tensor([0, 1, 1, 0])
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model.decode(onehots), ys)
        loss.backward()
        opt.step()
    assert model.decode(onehots).argmax(-1).tolist() == [0, 1, 1, 0]
