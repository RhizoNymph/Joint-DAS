# Feature: toy_tasks

Synthetic Phase-A tasks with known ground-truth causal structure, plus the toy
MLP model, its interchange-intervention site, and training/caching utilities.

## Scope

- Two tasks conforming to `jdas.types.Task`:
  - `HierarchicalEqualityTask` — `y = ((a==b) == (c==d))` over four unit-sphere
    symbol embeddings.
  - `BooleanCompositionTask` — `y = (x1 & x2) | x3` over three binary slots, each
    a fixed random embedding pair.
- The toy network `ToyMLP` and `MLPSite` (an `InterventionSite`).
- Toy-model training (`train_toy_model`) and cached load-or-train
  (`load_or_train_toy_model`).

## Non-scope

- The rotation, learned causal model, intervention machinery, trainers, eval
  (`src/jdas/{rotation,causal_model,intervention,training,eval}.py`) — owned
  elsewhere. This feature only produces the contracts they consume: batches,
  ground-truth variables/decoders, and a frozen intervention site.
- Phase B price-tagging / HF model wrappers.

## Data / control flow

### Task sampling

`sample_inputs(batch_size, generator) -> (inputs, labels)` produces a clean
supervised batch, used by the toy trainer.

`sample_batch(B, m, k_max, generator) -> InterventionBatch`:

1. Sample `base_inputs` `(B, input_dim)` and `source_inputs` `(B, m, input_dim)`
   with the same per-task sampler.
2. Compute `base_labels`/`source_labels` via `gt_label_fn(gt_variables(...))`.
3. Sample `source_assignment` `(B, k_max)` long in `[-1, m)` via
   `jdas.tasks._sampling.sample_source_assignment`:
   - `|I| ∈ {1, 2}` with prob `{0.5, 0.5}`, capped at `min(k_max, k_gt)` usable
     slots and at `m` (so distinct sources exist for `|I|=2`).
   - `|I|` distinct variable slots chosen uniformly among the first
     `min(k_max, k_gt)` slots.
   - distinct source indices assigned to swapped slots; all others `-1`.

Counterfactual semantics (used by core `L_cf` and by tests): take `base` GT
variables, replace slot `i` with `source_j`'s GT variable value whenever
`source_assignment[:, i] == j >= 0`, then apply `gt_label_fn`.

### Ground truth

- Hierarchical equality: `gt_variables` = `[a==b, c==d]` by exact vector
  comparison of the flattened input; `gt_label_fn(v) = (v[:,0]==v[:,1])`.
- Boolean composition: raw bits recovered by exact match against each slot's
  value-1 embedding; `gt_variables` = `[x1&x2, x3]`;
  `gt_label_fn(v) = v[:,0] | v[:,1]`.

### Model + site

`ToyMLP`: `n_layers` blocks of `Linear -> ReLU` then a linear head; width
`hidden` (=256, the site dimensionality `d`).

`MLPSite(model, layer_idx)`:
- freezes model params (`requires_grad_(False)`), `eval()` mode.
- `hidden(inputs)` = post-ReLU activations of block `layer_idx`, `(B, hidden)`,
  computed *without* `torch.no_grad` so downstream graphs can be built through a
  substituted hidden.
- `logits_with_hidden(inputs, hidden)` reruns blocks `layer_idx+1..` + head
  (the `inputs` arg is unused for the MLP but kept for protocol compatibility).
- `logits(inputs)` = full forward. Invariant:
  `logits_with_hidden(x, hidden(x)) == logits(x)` exactly.

### Training

`train_toy_model(task, device, steps, batch, lr, seed, ...)`: AdamW on fresh
`sample_inputs` batches; raises `ToyTrainingError` (structured) if final eval
accuracy `< target_acc` (default 0.99).

`load_or_train_toy_model(task, site_layer, device, cache_dir, seed, ...) ->
MLPSite`: caches `state_dict` (`.pt`) + metadata (`.json`) keyed by a sha256 of
`(task name, n_emb, input_dim, n_labels, seed, hidden, n_layers)`; trains + saves
on miss, loads on hit; returns an `MLPSite` at `site_layer`. **This exact
signature is imported by `experiments/run_phase_a.py`.**

## Files

- `src/jdas/tasks/__init__.py` — exports the two task classes.
- `src/jdas/tasks/hierarchical_equality.py` — `HierarchicalEqualityTask`,
  `TaskConfigError`.
- `src/jdas/tasks/boolean_comp.py` — `BooleanCompositionTask`.
- `src/jdas/tasks/_sampling.py` — `sample_source_assignment` (shared).
- `src/jdas/models/__init__.py` — exports model API.
- `src/jdas/models/toy.py` — `ToyMLP`, `MLPSite`, `train_toy_model`,
  `load_or_train_toy_model`, `ToyModelError`, `ToyTrainingError`.
- `tests/tasks/` — `test_tasks.py`, `test_toy_model.py`, `conftest.py`.

## Invariants / constraints

- Embeddings lie on the unit sphere in `R^n_emb` (norm 1, atol 1e-5).
- Hierarchical equality labels are ~50/50 balanced by construction; boolean
  composition has natural `P(y=1) = 0.625` under uniform bits (not balanced).
- `source_assignment` entries are in `[-1, m)`; only the first `min(k_max, k_gt)`
  slots are ever swapped; swapped sources are distinct when `|I|=2`.
- `gt_label_fn(gt_variables(inputs)) == labels` exactly.
- `MLPSite` model params are frozen; `hidden` keeps the autograd graph.
- Both tasks train to 100% at full spec (`hidden=256`, `steps=3000`); tests use
  reduced width/steps (`hidden=64`, `steps=800`) and run in a few seconds CPU.
