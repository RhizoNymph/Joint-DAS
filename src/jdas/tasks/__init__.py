"""Synthetic Phase-A tasks with known ground-truth causal structure.

Each task exposes the :class:`jdas.types.Task` protocol plus a couple of extra
conveniences used by the toy-model training loop and the core library's fixed
(baseline) causal model:

- ``sample_inputs(batch_size, generator) -> (inputs, labels)``: clean supervised
  batch for ordinary training of the toy network.
- ``gt_label_fn(vars) -> labels``: the ground-truth decoder mapping ground-truth
  variable values back to the task label (used by ``FixedCausalModel`` and by
  the counterfactual-semantics tests).
"""

from __future__ import annotations

from jdas.tasks.boolean_comp import BooleanCompositionTask
from jdas.tasks.hierarchical_equality import HierarchicalEqualityTask
from jdas.tasks.price_tagging import PriceTaggingTask

__all__ = [
    "BooleanCompositionTask",
    "HierarchicalEqualityTask",
    "PriceTaggingTask",
]
