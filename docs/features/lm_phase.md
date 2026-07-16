# Feature: lm_phase (Phase B — real HF language model)

## Scope

Run Joint-DAS on a frozen small HF instruct LM (Qwen2.5-0.5B/1.5B-Instruct)
using the Boundless-DAS **price-tagging** task, so ground-truth causal-variable
recovery is measurable on a real transformer.  Provides:

- the price-tagging task (`PriceTaggingTask`),
- a frozen-LM residual-stream intervention site (`HFSite`) + loader
  (`load_hf_site`),
- a featurized causal-model wrapper (`FeaturizedCausalModel`) so the learned
  causal model can read decoded `(X, Y, Z)` features rather than raw token ids,
- a zero-shot screening CLI (`screen_lm.py`) to pick model+template+layer,
- the Phase B runner (`run_phase_b.py`) mirroring `run_phase_a.py`.

## Non-scope

- No changes to the jdas core (`rotation`, `causal_model`, `intervention`,
  `training`, `eval`) — Phase B reuses `JointTrainer`/`DASTrainer`/`iia`/
  `recovery`/`refit_rotation` unchanged.
- No fine-tuning of the LM; weights are frozen throughout.
- No toy-model code (that is Phase A / `toy_tasks`).

## Data / control flow

1. `load_hf_site(model_name, layer, device)` loads a frozen `*ForCausalLM`
   (float32) + tokenizer (left padding) and wraps them as `HFSite`.
2. `PriceTaggingTask(tokenizer, template_id, device)` samples `(X, Y, Z)`,
   renders prompts via a template, tokenizes with **left padding** to a shared
   batch length `T`, and **packs** `input_ids` + `attention_mask` by stacking on
   a new axis: `base_inputs` is `(B, 2, T)`, `source_inputs` is `(B, m, 2, T)`.
   Ground truth: `L = (Z >= X)`, `U = (Z <= Y)`, label = `L AND U`.
3. The trainer samples interchange batches and calls:
   - `HFSite.hidden(inputs)` — forward under `no_grad`, capture the residual
     output of decoder layer `layer` at the **last** position (always the final
     prompt token thanks to left padding); returns detached `(B, d)`.
   - `interchange(...)` builds the mixed rotated hidden and calls
     `HFSite.logits_with_hidden(inputs, hidden)` — a second forward with a hook
     that replaces the last-position slice of the layer output with `hidden`;
     grads enabled, weights frozen, so the graph flows from the injected hidden
     through the upper layers/LM head to the yes/no logits.
   - Yes/No logits = `logsumexp` over candidate token-id variants of
     " yes"/"Yes"/... and " no"/"No"/..., producing `(B, 2) = [no, yes]`.
4. The learned causal model is a `FeaturizedCausalModel` (subclass of
   `LearnedCausalModel`) whose `_flatten` first applies
   `PriceTaggingTask.causal_features` — decoding token ids back to `(X, Y, Z)`
   via the tokenizer + regex, normalized to `[0, 1]` by `/10` — then runs the
   trainable per-variable MLP encoders + decoder on those 3 features. Feature
   extraction is deterministic and non-differentiable (fine: the trainable MLPs
   sit on top).
5. `run_phase_b.py` wires method → causal model:
   - `joint` / `random_rotation`: `FeaturizedCausalModel` (k_max variables),
     `JointTrainer`, then `recovery` + `refit_rotation` (joint only).
   - `das_true`: `FixedCausalModel(task.gt_variables, task.label_from_variables,
     k=2)` — layout `k_max = 2`.
   - `das_wrong`: single output-copy variable (`Z = y`) padded with dead
     always-zero variables up to `k_max` so `|I|=2` eval stays valid.
   Results are written as JSON with the same schema as Phase A (config, history,
   final metrics, and for learned methods recovery_matrix/best_assignment/
   recovery_score + refit_iia_1/2).

## Files

- `src/jdas/tasks/price_tagging.py` — `PriceTaggingTask`. Key methods:
  `sample_batch`, `sample_inputs`, `causal_features` (decode → `(B,3)`),
  `gt_variables` (`[L,U]`), `gt_label_fn`/`label_from_variables` (AND),
  `render_prompt`. Module `_TEMPLATES` holds the prompt variants.
- `src/jdas/models/hf.py` — `HFSite` (InterventionSite over a frozen LM),
  `load_hf_site` (loader, reads `HF_HOME`), `FeaturizedCausalModel`
  (LearnedCausalModel whose encoders read task features), `HFSiteError`.
- `experiments/screen_lm.py` — zero-shot screening CLI; writes
  `experiments/results/screen_<model>.json`.
- `experiments/run_phase_b.py` — Phase B runner CLI.
- `tests/lm/` — CPU-only tests (stub char tokenizer + tiny in-process Qwen2):
  `test_price_tagging.py`, `test_hf_site.py`, `conftest.py`.

## Invariants / constraints

- Left padding (`padding_side='left'`) is mandatory: `HFSite` reads position
  `-1`; it must be the final prompt token for every row.
- `base_inputs`/`source_inputs` are packed `(...,2,T)`: channel 0 = `input_ids`,
  channel 1 = `attention_mask`. Base and all sources in a `sample_batch` call
  share one padded `T` (tokenized jointly).
- `causal_features` must recover the exact 2-decimal `(X, Y, Z)` printed in the
  prompt (labels are computed from the rounded values, so decode == label
  source). It takes the **last** three decimal matches so few-shot example
  numbers earlier in the prompt (template 3) are ignored.
- The layout `k_max` must equal the causal model's variable count so the
  interchange assignment `(B, k_max)` validates: k_max for joint/random, 2 for
  das_true, and (padded) k_max for das_wrong.
- LM weights are frozen (`requires_grad_(False)`); only `Q`, subspace
  boundaries, and (for learned methods) the causal MLPs train.
- Labels are naturally ~1/3 "yes" (each Z region below/inside/above is ~1/3).

## GPU run commands (node1)

```
export HF_HOME=$HOME/hf-cache
# screening:
uv run python experiments/screen_lm.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --templates all --n 300 --device cuda --local-files-only
# a Phase B run (layer ~60% depth = 14 for 0.5B / 24 layers):
uv run python experiments/run_phase_b.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --layer 14 --method das_true --template-id <BEST> --device cuda \
    --steps 2000 --batch-size 32 --n-sources 2 --k-max 4 --v 2 \
    --local-files-only --out experiments/results/phase_b/<name>.json
```
