"""Toy models and their training loops (Phase A)."""

from __future__ import annotations

from jdas.models.toy import (
    MLPSite,
    ToyMLP,
    ToyModelError,
    ToyTrainingError,
    load_or_train_toy_model,
    train_toy_model,
)

__all__ = [
    "MLPSite",
    "ToyMLP",
    "ToyModelError",
    "ToyTrainingError",
    "load_or_train_toy_model",
    "train_toy_model",
]
