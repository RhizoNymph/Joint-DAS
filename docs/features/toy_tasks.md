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
signature is imported by the toy runner (`jdas run toy`).**

## Toy-model science baselines

Three falsification / measurement tools sit on top of the toy tasks. They share
one library, `src/jdas/hypotheses.py` (importable + unit-tested), which holds:
the named 2-input boolean function table (`BOOL_FNS`, `truth_table`,
`best_matching_fn`), the per-task 6-candidate `hypothesis_library`, and the
load-bearing `classify_solution` (see below).

### Wrong-composition baseline (`--method das_wrong_and`)

In the toy runner (`jdas run toy`). A **k=2** `FixedCausalModel` using the
task's TRUE ground-truth variables but a WRONG composition law:

- `hierarchical_equality`: `label = AND(E1, E2)` (truth: XNOR).
- `boolean_comp`: `label = XOR(A, x3)` (truth: OR).

Unlike the k=1 output-copy `das_wrong` strawman, this admits real `|I|=2`
multi-source interventions, so its counterfactual predictions must
systematically disagree with a faithful network. The run JSON records
`agreement_ceiling` (`{swap_size: fraction}`): over the task's own sampled
intervention distribution (~20k), the fraction of interventions where the
wrong-law counterfactual label equals the true-law label, computed with the
task's gt logic per swap size. This is the IIA a *perfect* DAS run with the
wrong law would approach **if N is faithful**. Interpretation: measured IIA near
the ceiling ⇒ falsification works; IIA well above the ceiling ⇒ something is off
(N unfaithful or vacuous alignment). Computed ceilings:
`hierarchical_equality ≈ 0.75` (AND vs XNOR agree on 3/4 atom combos);
`boolean_comp ≈ 0.87` (XOR vs OR agree except at `(A,x3)=(1,1)`).

### Discrete search baseline (`jdas run search`)

Enumerates all 15 unordered pairs of distinct candidates from the task's
`hypothesis_library`. For each pair it builds a k=2 `FixedCausalModel` whose
decoder is a **majority-label lookup** over the `(v1, v2)` combos, fit on ~8k
`sample_inputs` (unseen combos ⇒ global majority; each candidate model's clean
task accuracy is recorded), then trains `Q`+layout via `DASTrainer` and
evaluates held-out `iia_1`/`iia_2`. Emits JSON + a markdown ranking table
(pair, clean_task_acc, iia_1, iia_2, combined = mean of the two). Answers:
does brute-force search select `{E1,E2}` (or an equivalent basis), and does its
best IIA match the gradient-joint ~0.96? CLI: `--task --site-layer --seed
--steps (default 1500) --device --out`.

### Seed / basis variance study (`jdas run seed-study`)

For each seed, trains a joint run exactly as `jdas run toy` (same config
defaults), then classifies the learned solution WITHOUT retraining. It reuses
`_per_variable_effect` / task-loading from `experiments/introspect_toy.py`:
live variables = per-variable causal-effect rate `> 2%`; each live var's value
table over ~4096 fresh inputs is compared to `(E1, E2)` and to the named boolean
functions; `classify_solution` assigns one of:

- `atoms` — the (≤2) live vars ≈ `{E1, E2}` up to relabel (per-var cell-purity
  ≥ 0.9, covering both atoms).
- `equivalent_basis` — two live vars whose *joint* table has purity ≥ 0.9 and
  which jointly determine `(E1, E2)` **up to the E1↔E2 relabel symmetry** (three
  symmetry classes covered bijectively), e.g. `(OR, NAND)`.
- `output_copy` — a live var ≈ `y`/`¬y` and the remaining live vars don't
  complete a basis.
- `other` — degenerate / partial / >2 useful vars.

Records per seed: classification, each live var's best-matching function name,
`iia_1`, `iia_2`, `effective_k`, `recovery_score`; aggregates counts per class
and mean±std IIA. JSON + markdown. CLI: `--task hierarchical_equality
--site-layer 1 --seeds (default 10) --steps 4000 --device --out`. The classifier
is unit-tested (`tests/science/test_hypotheses.py`) on hand-built atoms /
`(OR,NAND)` / output-copy / degenerate cases — it is the load-bearing
measurement of the study.

## Files

- `src/jdas/hypotheses.py` — boolean function table, per-task hypothesis
  library, and `classify_solution` (toy-model science tooling).
- `src/jdas/cli/runners.py` (`jdas run toy`) — adds `das_wrong_and` +
  `agreement_ceiling`.
- `src/jdas/cli/runners.py` (`jdas run search`) — discrete search baseline.
- `src/jdas/cli/runners.py` (`jdas run seed-study`) — seed / basis variance study.
- `tests/science/test_hypotheses.py`, `tests/science/test_wrong_and_baseline.py`.
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
