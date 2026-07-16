"""Frozen HF causal-LM intervention site (Phase B).

:class:`HFSite` wraps a frozen HuggingFace ``*ForCausalLM`` and exposes the
**residual-stream output of one decoder layer at the last sequence position** as
an :class:`jdas.types.InterventionSite`.  The task is binary yes/no; the two
"logits" returned are ``[logit_no, logit_yes]`` where each is a ``logsumexp``
over a small set of tokenizer-specific token-id variants for the words
"yes"/"no".

Packed inputs
-------------
Inputs arrive packed as ``(B, 2, T)`` (see :mod:`jdas.tasks.price_tagging`):
channel 0 is ``input_ids`` and channel 1 is ``attention_mask``.  Because the
task uses left padding, position ``-1`` is always the final prompt token.

Gradient flow
-------------
- ``hidden`` runs under :func:`torch.no_grad` and returns a detached ``(B, d)``.
- ``logits_with_hidden`` re-runs the model with a forward hook on the target
  decoder layer that overwrites the last-position slice of the layer's output
  with the supplied ``hidden`` (which may carry grad).  This pass runs with
  gradients **enabled** so the autograd graph flows from the injected hidden,
  through the upper decoder layers and the LM head, to the yes/no logits.  Model
  parameters are frozen at construction (``requires_grad_(False)``) so no weight
  gradients accumulate.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from jdas.causal_model import LearnedCausalModel


class HFSiteError(ValueError):
    """Raised for invalid HF-site configuration or inputs."""


# Candidate surface forms whose token ids vote for each class.  We include a
# leading space (the common continuation form) and capitalized variants.
_YES_WORDS: tuple[str, ...] = (" yes", " Yes", "yes", "Yes", " YES", "YES")
_NO_WORDS: tuple[str, ...] = (" no", " No", "no", "No", " NO", "NO")


def _candidate_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    """First-token ids for each surface form, de-duplicated.

    A word maps to its *first* token id (the model predicts the answer token
    right after the prompt).  Multi-token forms contribute only their first
    token; duplicates are removed.
    """
    ids: list[int] = []
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.append(int(enc[0]))
    # De-duplicate preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise HFSiteError(f"no token ids resolved for words {words!r}")
    return out


class HFSite(nn.Module):
    """Residual-stream intervention site on a frozen HF causal LM.

    Args:
        model: a loaded ``*ForCausalLM`` (e.g. ``Qwen2ForCausalLM``).
        tokenizer: the matching tokenizer.
        layer: index of the decoder layer whose *output* residual stream is the
            site (``0 <= layer < n_layers``).
        yes_ids / no_ids: optional explicit token-id lists; if ``None`` they are
            derived from the tokenizer.
        device: device to run on.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer: int,
        yes_ids: list[int] | None = None,
        no_ids: list[int] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._layers = self._decoder_layers()
        if not (0 <= layer < len(self._layers)):
            raise HFSiteError(
                f"layer {layer} out of range [0, {len(self._layers)})"
            )

        self.no_ids = no_ids if no_ids is not None else _candidate_ids(tokenizer, _NO_WORDS)
        self.yes_ids = yes_ids if yes_ids is not None else _candidate_ids(tokenizer, _YES_WORDS)
        self._d = int(model.config.hidden_size)

    # -- protocol properties ------------------------------------------------

    @property
    def d(self) -> int:
        return self._d

    @property
    def n_labels(self) -> int:
        return 2

    # -- model plumbing -----------------------------------------------------

    def _decoder_layers(self) -> nn.ModuleList:
        """Return the list of decoder layer modules (``model.model.layers``)."""
        base = getattr(self.model, "model", self.model)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise HFSiteError("could not locate decoder layers at model.model.layers")
        return layers

    def _unpack(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack packed ``(B, 2, T)`` into ``(input_ids, attention_mask)``."""
        if inputs.dim() != 3 or inputs.shape[1] != 2:
            raise HFSiteError(
                f"expected packed (B, 2, T) inputs, got {tuple(inputs.shape)}"
            )
        ids = inputs[:, 0].to(self.device).long()
        mask = inputs[:, 1].to(self.device).long()
        return ids, mask

    @staticmethod
    def _layer_output(out) -> torch.Tensor:
        """Extract the hidden-states tensor from a decoder layer's output."""
        return out[0] if isinstance(out, tuple) else out

    def _yes_no_logits(self, lm_logits_last: torch.Tensor) -> torch.Tensor:
        """Map final-position vocab logits ``(B, V)`` to ``(B, 2)`` = [no, yes].

        Each class score is the ``logsumexp`` over that class's candidate token
        ids, so multiple surface forms contribute.
        """
        no_ids = torch.tensor(self.no_ids, device=lm_logits_last.device)
        yes_ids = torch.tensor(self.yes_ids, device=lm_logits_last.device)
        no = torch.logsumexp(lm_logits_last.index_select(-1, no_ids), dim=-1)
        yes = torch.logsumexp(lm_logits_last.index_select(-1, yes_ids), dim=-1)
        return torch.stack([no, yes], dim=-1)  # (B, 2)

    # -- InterventionSite interface ----------------------------------------

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Residual-stream output of ``layer`` at the last position ``(B, d)``.

        Runs under ``no_grad``; returns a detached tensor.
        """
        ids, mask = self._unpack(inputs)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inp, out):
            captured["h"] = self._layer_output(out)[:, -1, :].detach()

        handle = self._layers[self.layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=mask)
        finally:
            handle.remove()
        return captured["h"].to(self.device)

    def logits_with_hidden(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Re-run the model with ``hidden`` injected at (``layer``, last pos).

        Gradients are enabled so the graph flows from ``hidden`` up to the yes/no
        logits (model weights are frozen).
        """
        ids, mask = self._unpack(inputs)
        hidden = hidden.to(self.device)

        def hook(_module, _inp, out):
            h = self._layer_output(out)
            # Replace only the last-position slice; keep the rest of the graph.
            h = h.clone()
            h[:, -1, :] = hidden
            if isinstance(out, tuple):
                return (h, *out[1:])
            return h

        handle = self._layers[self.layer].register_forward_hook(hook)
        try:
            # Grads enabled (default); weights frozen at init.
            lm_out = self.model(input_ids=ids, attention_mask=mask)
        finally:
            handle.remove()
        return self._yes_no_logits(lm_out.logits[:, -1, :])

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Clean forward pass; return ``(B, 2)`` = [logit_no, logit_yes]."""
        ids, mask = self._unpack(inputs)
        with torch.no_grad():
            lm_out = self.model(input_ids=ids, attention_mask=mask)
        return self._yes_no_logits(lm_out.logits[:, -1, :])


def load_hf_site(
    model_name: str,
    layer: int,
    device: str | torch.device = "cpu",
    *,
    local_files_only: bool = False,
) -> HFSite:
    """Load a frozen HF causal LM (float32) and wrap it as an :class:`HFSite`.

    Reads the HF cache location from ``HF_HOME`` in the environment.  Loads in
    ``float32`` (0.5B/1.5B fit comfortably on a 24GB GPU and float32 avoids
    bf16 numerical noise in DAS training).

    Args:
        model_name: HF model id (e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``).
        layer: intervention decoder-layer index.
        device: device string.
        local_files_only: if ``True`` never hit the network (use cache only).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = os.environ.get("HF_HOME")
    hub_dir = os.path.join(cache_dir, "hub") if cache_dir else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=hub_dir,
        padding_side="left",
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=hub_dir,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    return HFSite(model, tokenizer, layer, device=device)


class FeaturizedCausalModel(LearnedCausalModel):
    """A :class:`LearnedCausalModel` whose encoders read task features, not ids.

    The trainer feeds raw packed token tensors to the causal model.  For a real
    LM those can't be flattened into a fixed-width vector, so this subclass
    overrides the input-flattening step to first apply a task feature callable
    (``feature_fn``) that maps packed ``(B, 2, T)`` inputs to a fixed ``(B,
    input_dim)`` feature matrix (e.g. normalized ``(X, Y, Z)``).  Feature
    extraction is non-differentiable, which is fine: the trainable per-variable
    MLPs and decoder sit on top of the features.

    Args:
        feature_fn: ``inputs -> (B, input_dim)`` float feature map.
        input_dim: dimensionality of the extracted features.
        (remaining args forwarded to :class:`LearnedCausalModel`.)
    """

    def __init__(
        self,
        feature_fn,
        input_dim: int,
        k_max: int,
        v: int = 2,
        n_labels: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, k_max, v=v, n_labels=n_labels, **kwargs)
        self._feature_fn = feature_fn

    def _flatten(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the feature map, then flatten to ``(B, input_dim)``."""
        feats = self._feature_fn(inputs)
        flat = feats.reshape(feats.shape[0], -1)
        if flat.shape[1] != self.input_dim:
            raise HFSiteError(
                f"feature_fn produced dim {flat.shape[1]}, expected {self.input_dim}"
            )
        return flat.to(next(self.parameters()).dtype)
