"""Toy models and their training loops (Phase A)."""

from __future__ import annotations

from jdas.models.hf import (
    FeaturizedCausalModel,
    HFSite,
    HFSiteError,
    load_hf_site,
)
from jdas.models.toy import (
    MLPSite,
    ToyMLP,
    ToyModelError,
    ToyTrainingError,
    load_or_train_toy_model,
    train_toy_model,
)

__all__ = [
    "FeaturizedCausalModel",
    "HFSite",
    "HFSiteError",
    "MLPSite",
    "ToyMLP",
    "ToyModelError",
    "ToyTrainingError",
    "load_hf_site",
    "load_or_train_toy_model",
    "train_toy_model",
]
