"""Fixtures for CPU-only Phase-B (LM) tests.

We avoid any network / real weights by building:

- :class:`CharTokenizer` -- a tiny char-level stub implementing exactly the
  tokenizer surface that :class:`PriceTaggingTask` and :class:`HFSite` use
  (``__call__`` with left padding, ``encode``/``decode``, ``padding_side``,
  ``pad_token``/``eos_token``).  Char-level round-trips the decimal numbers so
  ``causal_features`` regex decoding works.
- a tiny randomly-initialized ``Qwen2ForCausalLM`` (hidden 64, 4 layers) whose
  vocab matches the stub tokenizer.
"""

from __future__ import annotations

import pytest
import torch

# Character vocabulary: everything the templates + numbers can emit.  Index 0 is
# the pad token; index 1 is a catch-all/unknown+eos.
_CHARS = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,:?!\n-"
)


class CharTokenizer:
    """Minimal char-level tokenizer stub (left padding)."""

    def __init__(self) -> None:
        # id 0 = PAD, id 1 = EOS/UNK, then one id per character.
        self.pad_id = 0
        self.eos_id = 1
        self._itos = ["<pad>", "<eos>"] + list(_CHARS)
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.vocab_size = len(self._itos)
        self.padding_side = "left"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = self.pad_id
        self.eos_token_id = self.eos_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self._stoi.get(c, self.eos_id) for c in text]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            if skip_special_tokens and i in (self.pad_id, self.eos_id):
                continue
            out.append(self._itos[int(i)] if 0 <= int(i) < self.vocab_size else "")
        return "".join(out)

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        seqs = [self.encode(p, add_special_tokens=False) for p in prompts]
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            # left padding: place the sequence at the right end.
            ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, max_len - len(s):] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def tiny_model(tokenizer: CharTokenizer):
    """A tiny randomly-initialized Qwen2ForCausalLM matching the stub vocab."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model
