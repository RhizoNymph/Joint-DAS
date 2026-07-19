"""Toy MLP model, its interchange-intervention site, and training utilities.

The toy network is a plain ``Linear + ReLU`` stack followed by a linear head.
Its ``MLPSite`` wrapper exposes a single hidden vector (post-ReLU activations of
a chosen layer) as an interchange-intervention site implementing
:class:`jdas.types.InterventionSite`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from jdas.types import InterventionSite, Task


class ToyModelError(Exception):
    """Base error for toy-model construction/usage."""


class ToyTrainingError(ToyModelError):
    """Raised when a toy model fails to reach the required accuracy."""

    def __init__(self, task_name: str, accuracy: float, target: float, steps: int) -> None:
        self.task_name = task_name
        self.accuracy = accuracy
        self.target = target
        self.steps = steps
        super().__init__(
            f"toy model for task {task_name!r} reached only {accuracy:.4f} "
            f"eval accuracy after {steps} steps (target {target:.4f})"
        )


class ToyMLP(nn.Module):
    """A ``Linear + ReLU`` stack with a linear classification head.

    Architecture (``n_layers`` hidden blocks each ``Linear -> ReLU``):

        x -> [Linear(input_dim, hidden), ReLU]          (block 0)
          -> [Linear(hidden, hidden), ReLU] * (n_layers - 1)
          -> Linear(hidden, n_labels)                   (head)

    Args:
        input_dim: input feature dimension.
        hidden: hidden width ``d`` (also the intervention-site dimensionality).
        n_layers: number of ``Linear -> ReLU`` blocks (``>= 1``).
        n_labels: number of output classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        n_layers: int = 3,
        n_labels: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ToyModelError(f"n_layers must be >= 1, got {n_layers}")
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_labels = n_labels
        layers: list[nn.Linear] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, n_labels)

    def block(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply a single ``Linear -> ReLU`` block; return ``(B, hidden)``."""
        return torch.relu(self.layers[layer_idx](x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass; return ``(B, n_labels)`` logits."""
        for i in range(self.n_layers):
            x = self.block(x, i)
        return self.head(x)


@dataclass
class MLPSite(InterventionSite):
    """Interchange-intervention site at post-ReLU activations of one layer.

    The wrapped model's parameters are frozen (``requires_grad_(False)``). The
    site's hidden vector is the output of block ``layer_idx`` (after its ReLU),
    shape ``(B, hidden)``; ``logits_with_hidden`` reruns the remaining blocks and
    the head from that point. Because the weights are frozen, ``hidden`` has no
    trainable dependencies and is computed without ``torch.no_grad`` so callers
    can build autograd graphs through substituted hidden vectors.

    Args:
        model: the toy MLP to wrap.
        layer_idx: index of the block whose post-ReLU output is the site
            (``0 <= layer_idx < model.n_layers``).
    """

    model: ToyMLP
    layer_idx: int

    def __post_init__(self) -> None:
        if not (0 <= self.layer_idx < self.model.n_layers):
            raise ToyModelError(
                f"layer_idx {self.layer_idx} out of range "
                f"[0, {self.model.n_layers})"
            )
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def d(self) -> int:  # dimensionality of the site hidden vector
        return self.model.hidden

    @property
    def n_labels(self) -> int:
        return self.model.n_labels

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ``(B, hidden)`` post-ReLU activations of ``layer_idx``."""
        x = inputs
        for i in range(self.layer_idx + 1):
            x = self.model.block(x, i)
        return x

    def logits_with_hidden(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Rerun blocks ``layer_idx+1 ..`` and the head from ``hidden``.

        Args:
            inputs: unused for the MLP (kept for protocol compatibility).
            hidden: ``(B, hidden)`` substituted site activations.

        Returns:
            ``(B, n_labels)`` logits.
        """
        del inputs  # unused: the MLP site fully determines downstream from hidden
        x = hidden
        for i in range(self.layer_idx + 1, self.model.n_layers):
            x = self.model.block(x, i)
        return self.model.head(x)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full forward pass; return ``(B, n_labels)`` logits."""
        return self.model(inputs)


# -- training ----------------------------------------------------------------------


@torch.no_grad()
def _eval_accuracy(
    model: ToyMLP,
    task: Task,
    device: torch.device,
    generator: torch.Generator,
    n_batches: int = 8,
    batch: int = 512,
) -> float:
    """Estimate task accuracy over freshly sampled clean batches."""
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_batches):
        inputs, labels = task.sample_inputs(batch, generator)  # type: ignore[attr-defined]
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs).argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
    return correct / total


def train_toy_model(
    task: Task,
    device: torch.device,
    steps: int = 3000,
    batch: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 256,
    n_layers: int = 3,
    target_acc: float = 0.99,
) -> ToyMLP:
    """Train a :class:`ToyMLP` on ``task`` to at least ``target_acc`` accuracy.

    Uses AdamW and fresh batches from ``task.sample_inputs``.

    Args:
        task: task providing ``sample_inputs`` and ``input_dim``/``n_labels``.
        device: training device.
        steps: number of optimization steps.
        batch: batch size.
        lr: learning rate.
        seed: RNG seed for weights and data sampling.
        hidden: hidden width.
        n_layers: number of hidden blocks.
        target_acc: minimum eval accuracy required (raises otherwise).

    Returns:
        The trained ``ToyMLP`` in eval mode.

    Raises:
        ToyTrainingError: if the final eval accuracy is below ``target_acc``.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    input_dim = task.input_dim  # type: ignore[attr-defined]
    model = ToyMLP(
        input_dim=input_dim,
        hidden=hidden,
        n_layers=n_layers,
        n_labels=task.n_labels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
        inputs, labels = task.sample_inputs(batch, gen)  # type: ignore[attr-defined]
        inputs = inputs.to(device)
        labels = labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

    eval_gen = torch.Generator(device=device).manual_seed(seed + 12345)
    acc = _eval_accuracy(model, task, device, eval_gen, batch=batch)
    if acc < target_acc:
        raise ToyTrainingError(
            getattr(task, "name", type(task).__name__), acc, target_acc, steps
        )
    model.eval()
    return model


def _cache_key(
    task: Task, seed: int, hidden: int, n_layers: int
) -> tuple[str, dict]:
    """Return ``(hash_key, metadata)`` identifying a cached checkpoint."""
    meta = {
        "task": getattr(task, "name", type(task).__name__),
        "n_emb": getattr(task, "n_emb", None),
        "input_dim": task.input_dim,  # type: ignore[attr-defined]
        "n_labels": task.n_labels,
        "seed": seed,
        "hidden": hidden,
        "n_layers": n_layers,
    }
    blob = json.dumps(meta, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], meta


def load_or_train_toy_model(
    task: Task,
    site_layer: int,
    device: torch.device,
    cache_dir: str = "experiments/toy_ckpts",
    seed: int = 0,
    hidden: int = 256,
    n_layers: int = 3,
    steps: int = 3000,
    batch: int = 512,
    lr: float = 1e-3,
) -> MLPSite:
    """Load a cached toy model (or train + cache one) and return its site.

    The checkpoint (``state_dict`` + metadata JSON) is keyed by
    ``(task name, n_emb, seed, arch)``. If no matching checkpoint exists the
    model is trained via :func:`train_toy_model` and saved.

    Args:
        task: the task to train on.
        site_layer: layer index of the returned :class:`MLPSite`.
        device: device for training/inference.
        cache_dir: directory holding cached checkpoints.
        seed: training/data seed (also part of the cache key).
        hidden: hidden width.
        n_layers: number of hidden blocks.
        steps: training steps (used only when training).
        batch: training batch size (used only when training).
        lr: learning rate (used only when training).

    Returns:
        An :class:`MLPSite` wrapping the (frozen) model at ``site_layer``.
    """
    key, meta = _cache_key(task, seed, hidden, n_layers)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache / f"{meta['task']}_{key}.pt"
    meta_path = cache / f"{meta['task']}_{key}.json"

    if ckpt_path.exists():
        model = ToyMLP(
            input_dim=task.input_dim,  # type: ignore[attr-defined]
            hidden=hidden,
            n_layers=n_layers,
            n_labels=task.n_labels,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        model = train_toy_model(
            task,
            device,
            steps=steps,
            batch=batch,
            lr=lr,
            seed=seed,
            hidden=hidden,
            n_layers=n_layers,
        )
        torch.save(model.state_dict(), ckpt_path)
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2))

    return MLPSite(model=model, layer_idx=site_layer)
