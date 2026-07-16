"""Shared fixtures for task tests."""

from __future__ import annotations

import pytest
import torch

from jdas.tasks import BooleanCompositionTask, HierarchicalEqualityTask


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


@pytest.fixture
def heq() -> HierarchicalEqualityTask:
    return HierarchicalEqualityTask(n_emb=8)


@pytest.fixture
def boolean() -> BooleanCompositionTask:
    return BooleanCompositionTask(n_emb=8, seed=7)


ALL_TASKS = ["heq", "boolean"]
