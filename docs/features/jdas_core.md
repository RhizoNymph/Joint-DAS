# Feature: jdas_core

Core Joint-DAS library: orthogonal rotation + learned subspace layout, the
learned/fixed high-level causal model, the interchange-intervention machinery,
the joint / classic-DAS trainers, and the evaluation metrics. Implements the
method in `docs/DESIGN.md`.

## Scope

- Rotation `Q` (orthogonal) and learned subspace boundaries over rotated dims.
- Learned causal model `H_theta` (per-variable encoders + decoder, straight-
  through argmax discretization) and a non-trainable `FixedCausalModel` for
  baselines and freeze-and-refit.
- Interchange interventions: swap aligned-subspace content of a base hidden with
  chosen source hiddens, rerun the frozen network `N`.
- Trainers: `JointTrainer` (learn `Q`, boundaries, `H`), `DASTrainer` (classic
  DAS, `H` fixed), `refit_rotation` (freeze-and-refit protocol), random-rotation
  control.
- Eval: IIA per swap size, recovery matrix vs. ground truth, effective-k.
- Phase A CLI entry point (`experiments/run_phase_a.py`).

## Non-scope

- Concrete tasks (`src/jdas/tasks/*`) and toy/HF models (`src/jdas/models/*`):
  authored separately; core codes only against the `Task` / `InterventionSite`
  protocols in `src/jdas/types.py`.
- Phase B (real LM) specifics live in the LM feature.

## Data / control flow

1. A `Task.sample_batch` yields an `InterventionBatch` (base + `m` source inputs
   + labels). The trainer replaces `source_assignment` with a sampled swap
   (|I| in {1,2}, distinct sources for |I|=2).
2. `interchange(site, rotation, layout, batch)`:
   - `r_b = rotation.rotate(site.hidden(base))` -> `(B, d)`.
   - `r_s = rotation.rotate(site.hidden(sources_flat))` -> `(B, m, d)`
     (single `(B*m, ...)` forward).
   - masks `(k_max, d)` from `layout` (soft for train, hard for eval); per
     example, for each swapped variable `i` gather source `j_i`'s rotated hidden
     and overwrite the masked coordinates.
   - unrotate and call `site.logits_with_hidden(base, h_new)` -> `(B, n_labels)`.
   Gradients flow to `Q` (both base and source paths — sources are not detached)
   and to the boundary parameters (through the soft masks). Site weights are
   frozen by the site implementation.
3. `H.counterfactual_predict(base, sources, assignment)` computes the high-level
   counterfactual label logits with the same swap semantics on discrete
   variables.
4. Losses (`training.py._compute_losses`):
   - `L_cf` = CE(N's intervened log-probs, H's counterfactual straight-through
     one-hot target) — grads reach `H` through the ST estimator and reach `Q`/
     boundaries through the intervention.
   - symmetric term (weight `lambda_cf_symmetric=0.5`) = CE(H counterfactual
     logits, N's hard intervened label) — trains H's encoders from N's behavior.
   - `L_task` = CE(H.predict, true label) on clean base + source inputs.
   - `L_sparse` = `lambda_sparse * total_aligned_dims / d` when
     `sparse_mode="normalized"` (default), or `lambda_sparse *
     total_aligned_dims` when `sparse_mode="per_dim"` (unnormalized, so the
     per-dimension gradient is `lambda_sparse` itself — bites at large `d` where
     the normalized `lambda/d` is negligible, e.g. `d=1536`).
5. AdamW updates rotation + layout (+ H for joint). Temperatures (`tau_g` for the
   ST softmax, `tau_m` for masks) anneal `start -> end` (linear/cosine) over
   `steps`.
6. Eval (`eval.py`, all `no_grad`, hard masks + hard variables): IIA per swap
   size, recovery matrix + best assignment, effective-k.

## Orientation / conventions

- Rotation: `rotate(h) = h @ Q.T` (== `Linear(h)`); `unrotate(r) = r @ Q`;
  `unrotate(rotate(h)) == h`. Aligned coordinate `p` is `h @ Q[p]` (row `p`).
- Subspace: widths `w_i = softplus(raw_i) >= 0` (unbounded, default) or `w_i =
  max_width * sigmoid(raw_i) < max_width` (bounded, when `max_width` is set —
  a hard per-variable width cap that forbids a single variable owning the whole
  space); boundaries `c_i = sum_{j<=i} w_j` clamped to `d`; variable `i` owns
  rotated dims `[c_i, c_{i+1})`. Blocks are cumulative and disjoint by
  construction, so masks never overlap.
- `source_assignment`: `(B, k_max)` long in `[-1, m)`; `-1` = keep base, `j>=0` =
  take variable `i` (and its subspace) from source `j`. (Matches `types.py`.)

## Files

- `src/jdas/rotation.py` — `OrthogonalRotation(d, freeze=)` (orthogonal-
  parametrized `Linear`, `rotate`/`unrotate`/`set_matrix`), `SubspaceLayout(d,
  k_max, init_width, max_width=None, min_temp, max_temp)` (`widths`,
  `soft_masks`, `hard_masks`, `hard_widths`, `total_aligned_dims`,
  `set_temperature`). `max_width` selects the bounded sigmoid parameterization
  (requires `init_width < max_width`). Exception: `RotationError`.
- `src/jdas/causal_model.py` — `LearnedCausalModel` (encoders `g_i`, decoder,
  `variables`/`predict`/`counterfactual_predict`, `set_temperature`),
  `FixedCausalModel(gt_variables_fn, label_fn, k, v, n_labels)`. Straight-through
  helper `_straight_through_onehot`. Exception: `CausalModelError`.
- `src/jdas/intervention.py` — `interchange(site, rotation, layout, batch,
  hard=)`. Exception: `InterventionError`.
- `src/jdas/training.py` — `JointConfig` (dataclass; includes `sparse_mode`
  `"normalized"|"per_dim"`), `JointTrainer`, `DASTrainer`, `refit_rotation`,
  `save_checkpoint(path, rotation, layout, causal_model, config, extra)` /
  `load_checkpoint(path, feature_fn=None, map_location=)` (torch.save of
  state_dicts + JSON meta carrying `d`, layout `k_max`/`max_width`/temps, and
  learned-model `input_dim`/`v`/`n_labels`/encoder+decoder widths;
  `FixedCausalModel` has no state and is rebuilt by the caller). Exceptions:
  `TrainingError`, `CheckpointError`.
- `src/jdas/eval.py` — `iia`, `recovery` (+ `RecoveryResult`), `effective_k`.
  Exception: `EvalError`.
- `src/jdas/__init__.py` — public re-exports.
- `experiments/run_phase_a.py` — argparse CLI; lazy-imports tasks/models
  (`jdas.tasks.*`, `jdas.models.toy.load_or_train_toy_model`); writes results
  JSON (config + history + final metrics + recovery + refit IIA).
- `tests/core/` — `fakes.py` (inline `XorTask`, `MLPSite`, `IdentitySite`,
  `RiggedSite`), `test_rotation.py`, `test_causal_model.py`,
  `test_intervention.py`, `test_eval.py`, `test_training.py`.

## Invariants and constraints

- `Q @ Q.T == I` for all parameter values (orthogonal parametrization),
  preserved across optimizer steps.
- Subspace blocks are contiguous, disjoint, and inside `[0, d]`. Soft-mask column
  sums `<= 1`. Soft masks -> hard masks as `tau_m -> 0`. With `max_width` set,
  every hard/soft width `<= max_width` for any parameter value.
- Checkpoint round-trip is exact: reloaded `Q`, hard widths, and causal-model
  predictions match the saved modules (validated by tests).
- `variables()` are exact one-hots in the forward pass; gradients follow
  `softmax(logits / tau_g)`.
- `interchange` must not detach source hiddens from `Q`.
- All eval is deterministic given a `torch.Generator` and runs under `no_grad`
  with hard discretization.
- `source_assignment` entries lie in `[-1, m)`; validated by `_check_assignment`.

## Method interface expected of collaborators

- `Task`: `n_labels`, `k_gt`, `sample_batch(...) -> InterventionBatch`,
  `gt_variables(inputs) -> (B, k_gt)`. For `das_true`/`das_wrong` baselines the
  Phase A runner also expects `task.label_from_variables(vars) -> (B,)`.
- `InterventionSite`: `d`, `n_labels`, `hidden`, `logits_with_hidden`, `logits`
  (frozen weights, graph kept from a substituted hidden to logits).
- `jdas.models.toy.load_or_train_toy_model(task, site_layer, device,
  cache_dir="experiments/toy_ckpts") -> InterventionSite`.
