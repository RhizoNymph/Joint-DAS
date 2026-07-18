"""Price-tagging task (Phase B, Boundless-DAS style).

The prompt asks whether an item's price ``Z`` falls between a lower bound ``X``
and an upper bound ``Y``:

    "Please say yes only if it costs between X.xx and Y.yy dollars,
     otherwise no. Item: Z.zz dollars" -> Yes / No

Sampling (2 decimals throughout)
--------------------------------
- ``X ~ U[0.50, 8.00]``.
- ``Y = X + U[1.00, 9.99 - X]`` so ``X < Y`` always (upper bound below 9.99).
- ``Z`` is sampled so the three regions ``Z < X`` / ``X <= Z <= Y`` /
  ``Z > Y`` each occur with probability ``1/3``.  Consequently the label
  ``yes`` (in range) has probability ~``1/3`` and ``no`` ~``2/3`` -- this is the
  natural distribution of the task and is intentional (documented here).

Ground-truth causal variables (``k_gt = 2``)
--------------------------------------------
``L = (Z >= X)`` (lower bound satisfied) and ``U = (Z <= Y)`` (upper bound
satisfied).  The label is ``L AND U``.  Same two-boolean skeleton as
hierarchical equality, so recovery is measurable on a real LM.

Input packing
-------------
:class:`jdas.types.InterventionBatch` requires a single tensor for
``base_inputs``.  A tokenized prompt needs both ``input_ids`` and an
``attention_mask`` (left padding).  We pack them by stacking along a new axis::

    packed[b] = stack([input_ids[b], attention_mask[b]])   # (2, T)

so ``base_inputs`` has shape ``(B, 2, T)`` and ``source_inputs`` has shape
``(B, m, 2, T)``.  :class:`jdas.models.hf.HFSite` unpacks channel 0 as
``input_ids`` and channel 1 as ``attention_mask``.  Left padding guarantees the
**last** sequence position is always the final prompt token.

Feature extraction for the causal model
---------------------------------------
The learned causal model can't read variable-length token ids directly, so
:meth:`causal_features` decodes the token ids back to the three floats
``(X, Y, Z)`` (via the tokenizer + a regex on the rendered prompt) and
normalizes to ``[0, 1]`` by dividing by 10.  This is a deterministic,
non-differentiable feature map; the trainable per-variable MLPs sit on top of
it (see :mod:`jdas.models.hf` / ``jdas run lm``).
"""

from __future__ import annotations

import re

import torch

from jdas.tasks._sampling import sample_source_assignment
from jdas.types import InterventionBatch


class PriceTaggingError(ValueError):
    """Raised for invalid price-tagging configuration or undecodable inputs."""


# Prompt templates.  ``{lo}``, ``{hi}``, ``{item}`` are formatted with 2-decimal
# strings.  ``template_id`` selects one of these; screening picks the best.
_TEMPLATES: tuple[str, ...] = (
    # 0: terse instruction, no few-shot.
    "Please say yes only if it costs between {lo} and {hi} dollars, "
    "otherwise no.\nItem: {item} dollars",
    # 1: question phrasing.
    "Does an item that costs {item} dollars fall between {lo} and {hi} dollars? "
    "Answer yes or no.",
    # 2: explicit range phrasing.
    "The valid price range is {lo} to {hi} dollars. "
    "An item costs {item} dollars. Is its price in the valid range?",
    # 3: with one few-shot example baked into the plain prompt.
    "Say yes if the price is within the range, otherwise no.\n"
    "Range: 2.00 to 5.00 dollars. Item: 3.50 dollars. Answer: yes\n"
    "Range: {lo} to {hi} dollars. Item: {item} dollars. Answer:",
)

# Regex to pull the three 2-decimal numbers out of a rendered prompt, in the
# order lo, hi, item.  All templates above emit exactly lo, hi, item in order
# (the few-shot example in template 3 uses 2.00/5.00/3.50 first -> we take the
# LAST three matches so the example numbers are skipped).
_NUM_RE = re.compile(r"\d+\.\d{2}")


def _fmt(x: float) -> str:
    """Format a price as a fixed 2-decimal string."""
    return f"{x:.2f}"


class PriceTaggingTask:
    """Boundless-DAS price-tagging task over a real HF tokenizer.

    Args:
        tokenizer: an HF tokenizer (must support ``__call__`` with
            ``padding``/``return_tensors`` and ``apply_chat_template`` for chat
            templates; a stub implementing the same surface is fine for tests).
        template_id: index into the prompt template table.
        device: device for the produced tensors.
        use_chat_template: if ``True`` render the question through the
            tokenizer's chat template with an ``"Answer:"`` assistant prefix;
            else use the plain rendered string.
        position: intervention-position mode passed to the site.  ``"last"``
            (default) captures/injects at the final sequence position (the
            historical behavior; produces packed ``(B, 2, T)``).  ``"z_digits"``
            captures/injects at the **last token of the item amount ``Z``** in
            each prompt, producing packed ``(B, 3, T)`` with a per-example
            position channel (see :mod:`jdas.models.hf`).
    """

    n_labels: int = 2
    k_gt: int = 2

    def __init__(
        self,
        tokenizer,
        template_id: int = 0,
        device: str | torch.device = "cpu",
        *,
        use_chat_template: bool = False,
        position: str = "last",
    ) -> None:
        if not (0 <= template_id < len(_TEMPLATES)):
            raise PriceTaggingError(
                f"template_id {template_id} out of range [0, {len(_TEMPLATES)})"
            )
        if position not in ("last", "z_digits"):
            raise PriceTaggingError(
                f"position must be 'last' or 'z_digits', got {position!r}"
            )
        self.tokenizer = tokenizer
        self.template_id = template_id
        self.device = torch.device(device)
        self.use_chat_template = use_chat_template
        self.position = position
        self.name = "price_tagging"
        # Feature dim consumed by the learned causal model: (X, Y, Z) normalized.
        self.input_dim = 3
        if getattr(tokenizer, "padding_side", None) != "left":
            # Left padding is required so the last position is the final token.
            tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

    # -- ground-truth causal model ------------------------------------------

    @staticmethod
    def gt_label_fn(vars: torch.Tensor) -> torch.Tensor:
        """Ground-truth decoder: ``L AND U`` (label 1 == in range).

        Args:
            vars: ``(B, 2)`` long, entries ``[L=(Z>=X), U=(Z<=Y)]``.

        Returns:
            ``(B,)`` long task labels.
        """
        return (vars[:, 0].bool() & vars[:, 1].bool()).long()

    # Alias so the LM runner (mirroring the toy runner) can build the
    # true/wrong FixedCausalModel via ``task.label_from_variables``.
    def label_from_variables(self, vars: torch.Tensor) -> torch.Tensor:
        return self.gt_label_fn(vars)

    # -- prompt rendering ---------------------------------------------------

    def render_prompt(self, lo: float, hi: float, item: float) -> str:
        """Render one prompt string for the current template (no chat wrap)."""
        return _TEMPLATES[self.template_id].format(
            lo=_fmt(lo), hi=_fmt(hi), item=_fmt(item)
        )

    def _wrap_prompt(self, text: str) -> str:
        """Apply the chat template (with an ``Answer:`` assistant prefix) or not."""
        if not self.use_chat_template:
            return text
        messages = [{"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return rendered + "Answer:"

    # -- sampling -----------------------------------------------------------

    def _sample_xyz(
        self, n: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``(X, Y, Z)`` triples (each ``(n,)`` float, 2 decimals).

        Region of ``Z`` (below / inside / above) is uniform over the three
        options so labels are ~1/3 yes.
        """
        u = lambda: torch.rand(n, generator=generator, device=device)  # noqa: E731
        x = 0.50 + u() * (8.00 - 0.50)
        y = x + 1.00 + u() * (9.99 - x - 1.00)
        # region: 0 below X, 1 inside [X, Y], 2 above Y -- each prob 1/3.
        region = torch.randint(0, 3, (n,), generator=generator, device=device)
        z = torch.empty(n, device=device)
        below = region == 0
        inside = region == 1
        above = region == 2
        # Below: U[0, X). Inside: U[X, Y]. Above: U(Y, 9.99].
        z[below] = x[below] * u()[below]
        z[inside] = x[inside] + u()[inside] * (y[inside] - x[inside])
        z[above] = y[above] + u()[above] * (9.99 - y[above])
        # Round to 2 decimals (the rendered prompt only shows 2 decimals; the
        # causal features decode the rounded values, so round here for
        # consistency between labels and decoded features).
        x = torch.round(x * 100) / 100
        y = torch.round(y * 100) / 100
        z = torch.round(z * 100) / 100
        return x, y, z

    def _tokenize(
        self,
        prompts: list[str],
        device: torch.device,
        unpadded_positions: list[int] | None = None,
    ) -> torch.Tensor:
        """Tokenize prompts with left padding; return packed ``(N, 2, T)`` or
        ``(N, 3, T)``.

        Channel 0 = ``input_ids``, channel 1 = ``attention_mask``.  When
        ``unpadded_positions`` is provided (an index into each prompt's *own*
        token sequence) a third channel is added holding the corresponding
        **left-padding-adjusted** position broadcast across the time axis: for a
        prompt of ``L`` tokens padded to ``T``, unpadded index ``j`` maps to
        ``T - L + j``.
        """
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if unpadded_positions is None:
            return torch.stack([ids, mask], dim=1)  # (N, 2, T)

        t = ids.shape[-1]
        # Under left padding each prompt occupies the rightmost L = sum(mask)
        # columns, so unpadded index j lands at column (T - L) + j.
        lengths = mask.sum(dim=1)  # (N,) tokens per prompt
        offsets = t - lengths  # (N,) left-pad width
        pos = offsets + torch.tensor(
            unpadded_positions, device=device, dtype=offsets.dtype
        )
        pos = pos.clamp_(0, t - 1)
        pos_channel = pos.view(-1, 1).expand(-1, t)  # (N, T) broadcast
        return torch.stack([ids, mask, pos_channel], dim=1)  # (N, 3, T)

    def _z_final_token_index(self, prompt: str, item: float) -> int:
        """Index (into ``prompt``'s own token sequence) of the last token of ``Z``.

        Located robustly across tokenizers: find the character span of the last
        occurrence of the rendered ``Z`` string in ``prompt`` (the item amount
        always appears after lo/hi), then tokenize the prefix ending at that span
        and return ``len(prefix_tokens) - 1``.  Falls back to the final token if
        the substring cannot be located.
        """
        z_str = _fmt(item)
        end = prompt.rfind(z_str)
        if end < 0:
            enc = self.tokenizer.encode(prompt, add_special_tokens=False)
            return max(0, len(enc) - 1)
        end += len(z_str)  # char index just past Z
        prefix = prompt[:end]
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        return max(0, len(prefix_ids) - 1)

    def _render_batch(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Render + tokenize a batch of triples to packed ``(N, 2, T)`` tensors,
        or ``(N, 3, T)`` when ``position == "z_digits"``.
        """
        prompts = [
            self._wrap_prompt(self.render_prompt(float(x[i]), float(y[i]), float(z[i])))
            for i in range(x.shape[0])
        ]
        if self.position == "z_digits":
            positions = [
                self._z_final_token_index(prompts[i], float(z[i]))
                for i in range(x.shape[0])
            ]
            return self._tokenize(prompts, device, unpadded_positions=positions)
        return self._tokenize(prompts, device)

    def _labels_from_xyz(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute task labels ``(N,)`` from float triples."""
        lo_ok = (z >= x).long()
        hi_ok = (z <= y).long()
        return self.gt_label_fn(torch.stack([lo_ok, hi_ok], dim=-1))

    def sample_inputs(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean supervised batch: ``(packed_inputs (B,2,T), labels (B,))``."""
        device = self.device
        x, y, z = self._sample_xyz(batch_size, generator, device)
        inputs = self._render_batch(x, y, z, device)
        labels = self._labels_from_xyz(x, y, z)
        return inputs, labels

    def sample_batch(
        self,
        batch_size: int,
        n_sources: int,
        k_max: int,
        generator: torch.Generator,
    ) -> InterventionBatch:
        """Sample an interchange-intervention batch (see :class:`InterventionBatch`).

        ``base_inputs`` is ``(B, 2, T)`` and ``source_inputs`` is ``(B, m, 2, T)``
        where ``T`` is the per-call max prompt length (base and sources are
        tokenized together so they share one padded length).
        """
        device = self.device
        m = n_sources
        # Sample base + all sources' triples together, then tokenize jointly so
        # base and sources share the same padded length T.
        n_total = batch_size * (1 + m)
        x, y, z = self._sample_xyz(n_total, generator, device)
        packed = self._render_batch(x, y, z, device)  # (B*(1+m), C, T), C in {2,3}
        labels = self._labels_from_xyz(x, y, z)  # (B*(1+m),)

        c, t = packed.shape[1], packed.shape[-1]
        packed = packed.reshape(batch_size, 1 + m, c, t)
        base_inputs = packed[:, 0]  # (B, C, T)
        source_inputs = packed[:, 1:]  # (B, m, C, T)
        labels = labels.reshape(batch_size, 1 + m)
        base_labels = labels[:, 0]  # (B,)
        source_labels = labels[:, 1:]  # (B, m)

        source_assignment = sample_source_assignment(
            batch_size, n_sources, k_max, self.k_gt, generator, device
        )
        return InterventionBatch(
            base_inputs=base_inputs,
            source_inputs=source_inputs,
            source_assignment=source_assignment,
            base_labels=base_labels,
            source_labels=source_labels,
        )

    # -- feature extraction / ground truth ----------------------------------

    def causal_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Decode packed token ids back to normalized ``(X, Y, Z)`` features.

        Args:
            inputs: packed ``(B, 2, T)`` (channel 0 = input_ids,
                channel 1 = attention_mask).

        Returns:
            ``(B, 3)`` float in ``[0, 1]``: ``[X, Y, Z] / 10``.  Deterministic.
        """
        if inputs.dim() != 3 or inputs.shape[1] not in (2, 3):
            raise PriceTaggingError(
                f"causal_features expects packed (B, 2, T) or (B, 3, T), got "
                f"{tuple(inputs.shape)}"
            )
        ids = inputs[:, 0]  # (B, T)
        mask = inputs[:, 1]  # (B, T)
        b = ids.shape[0]
        feats = torch.empty(b, 3, device=inputs.device)
        for i in range(b):
            # Drop left padding using the attention mask before decoding.
            valid = ids[i][mask[i].bool()]
            text = self.tokenizer.decode(valid.tolist(), skip_special_tokens=True)
            nums = _NUM_RE.findall(text)
            if len(nums) < 3:
                raise PriceTaggingError(
                    f"expected >=3 decimal numbers in decoded prompt, got "
                    f"{len(nums)}: {text!r}"
                )
            # Take the LAST three matches so any few-shot example numbers earlier
            # in the prompt are ignored (they always precede lo/hi/item).
            lo, hi, item = (float(v) for v in nums[-3:])
            feats[i, 0] = lo / 10.0
            feats[i, 1] = hi / 10.0
            feats[i, 2] = item / 10.0
        return feats

    def gt_variables(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recompute ground-truth ``[L=(Z>=X), U=(Z<=Y)]`` from features.

        Args:
            inputs: packed ``(B, 2, T)``.

        Returns:
            ``(B, 2)`` long, entries ``[L, U]``.
        """
        feats = self.causal_features(inputs)  # (B, 3) normalized by /10
        x, y, z = feats[:, 0], feats[:, 1], feats[:, 2]
        lo_ok = (z >= x).long()
        hi_ok = (z <= y).long()
        return torch.stack([lo_ok, hi_ok], dim=-1)
