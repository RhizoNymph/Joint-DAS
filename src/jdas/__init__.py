"""Joint-DAS core library.

Learning causal representations jointly with distributed alignment search:
an orthogonal rotation + learned subspace layout aligns a frozen network's
hidden state to discrete high-level variables that are themselves learned
(or hand-specified for baselines).
"""

from __future__ import annotations

from .causal_model import CausalModelError, FixedCausalModel, LearnedCausalModel
from .eval import RecoveryResult, effective_k, iia, recovery
from .intervention import InterventionError, interchange
from .rotation import OrthogonalRotation, RotationError, SubspaceLayout
from .training import (
    DASTrainer,
    JointConfig,
    JointTrainer,
    TrainingError,
    refit_rotation,
)
from .types import InterventionBatch, InterventionSite, Task

__all__ = [
    "CausalModelError",
    "DASTrainer",
    "FixedCausalModel",
    "InterventionBatch",
    "InterventionError",
    "InterventionSite",
    "JointConfig",
    "JointTrainer",
    "LearnedCausalModel",
    "OrthogonalRotation",
    "RecoveryResult",
    "RotationError",
    "SubspaceLayout",
    "Task",
    "TrainingError",
    "effective_k",
    "interchange",
    "iia",
    "recovery",
    "refit_rotation",
]
