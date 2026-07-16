Overview:
  description: >
    Research codebase for Joint-DAS: an extension of Distributed Alignment
    Search that learns the high-level causal model jointly with the orthogonal
    rotation/alignment, instead of requiring a hand-proposed causal model.
    See docs/DESIGN.md for the method.
  subsystems:
    - jdas core (src/jdas): rotation + subspace masks, learned causal model,
      interchange intervention machinery, joint/DAS trainers, evaluation.
    - tasks (src/jdas/tasks): synthetic tasks with known ground-truth causal
      structure + LM prompt tasks; counterfactual pair generators.
    - models (src/jdas/models): toy networks (MLP/transformer) and their
      training; HF model wrappers for Phase B.
    - experiments (experiments/): config-driven entry points writing JSON
      results.
    - scripts (scripts/): GPU node sync/launch/collect (node0/1/2, 1x3090 each).
  data_flow: >
    Task generates (base, sources, intervention-spec, labels) batches →
    intervention machinery runs frozen network N with rotated-subspace swaps →
    trainer compares N's intervened output with learned causal model H's
    counterfactual prediction (L_cf) + H's task fit (L_task) → gradients update
    rotation Q, subspace boundaries, and H's parameters. Eval computes held-out
    IIA and recovery of ground-truth variables.

Features Index:
  jdas_core:
    description: Rotation, learned causal model, interventions, trainers, eval.
    entry_points: [src/jdas/training.py, src/jdas/eval.py]
    depends_on: []
    doc: docs/features/jdas_core.md
  toy_tasks:
    description: Synthetic ground-truth tasks + toy model training (Phase A).
    entry_points: [src/jdas/tasks/, src/jdas/models/]
    depends_on: [jdas_core]
    doc: docs/features/toy_tasks.md
  lm_phase:
    description: Phase B on a small HF LM (price tagging task).
    entry_points: [src/jdas/tasks/price_tagging.py, experiments/]
    depends_on: [jdas_core]
    doc: docs/features/lm_phase.md
