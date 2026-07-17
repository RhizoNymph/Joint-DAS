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
      results (run_phase_a, run_phase_b, screen_lm, introspect_phase_a) plus
      analyze.py / analyze_night2.py (aggregate JSON -> summary md + docs/assets
      plots). Phase-A "science" tooling: seed_study.py (basis variance over
      seeds), search_baseline.py (brute-force candidate-pair enumeration),
      and the das_wrong_and falsification baseline, all sharing
      src/jdas/hypotheses.py (boolean fn library + solution classifier).
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
    entry_points: [src/jdas/rotation.py, src/jdas/causal_model.py,
      src/jdas/intervention.py, src/jdas/training.py, src/jdas/eval.py,
      experiments/run_phase_a.py]
    depends_on: []
    doc: docs/features/jdas_core.md
  toy_tasks:
    description: Synthetic ground-truth tasks + toy model training (Phase A).
    entry_points: [src/jdas/tasks/, src/jdas/models/]
    depends_on: [jdas_core]
    doc: docs/features/toy_tasks.md
  lm_phase:
    description: >
      Phase B on a small HF LM (price tagging task). Screened Qwen2.5-1.5B-Instruct
      (0.5B was degenerate) at template 3 plain, ~81% zero-shot; runs at layer 17.
      Night 2: collapse mechanism fixed via per-dim sparsity + hard width cap
      (run_phase_b flags --sparse-mode per_dim / --max-width / --init-width);
      capped joint (k_eff 4, iia_1 0.855) beats the capped control (k_eff 0).
    entry_points: [src/jdas/tasks/price_tagging.py, src/jdas/models/hf.py,
      experiments/run_phase_b.py, experiments/screen_lm.py]
    depends_on: [jdas_core]
    doc: docs/features/lm_phase.md
  phase_a_science:
    description: >
      Night-2 measurement tooling over the two GT boolean atoms: seed/basis
      variance study, brute-force discrete search baseline, and the k=2
      wrong-composition (das_wrong_and) falsification with analytic agreement
      ceilings. Establishes basis non-identifiability (E1+E2 not a valid
      alignment at deep sites) from two independent methods.
    entry_points: [experiments/seed_study.py, experiments/search_baseline.py,
      experiments/run_phase_a.py, src/jdas/hypotheses.py]
    depends_on: [jdas_core, toy_tasks]
    doc: docs/features/jdas_core.md
