# Variable gates (L0 minimality)

## Scope

An intrinsic Occam mechanism for joint-DAS: per-variable stochastic gates with
an L0-style cost, so the number of *live* causal variables is learned rather
than fixed at `k_max`. Closes the wave-C gap (RESULTS.md N2.6): width-sparsity
penalizes dims, not variables, so joint prefers redundant 4-variable solutions
even when 2 suffice.

Non-scope: pruning *dims within* a variable (existing width caps / per-dim
sparsity handle that); gating for the fixed-H baselines (das_true/das_wrong
keep their exact hypothesis).

## Mechanism

Hard-concrete gates (Louizos et al. 2017, "Learning Sparse Neural Networks
through L0 Regularization"), one gate per variable slot:

- Parameter: `log_alpha ∈ R^{k_max}`, init `+2.0` (≈0.88 open) so training
  starts with all variables available.
- Constants: `beta = 2/3`, `gamma = -0.1`, `zeta = 1.1`.
- Train-time sample: `u ~ U(0,1)`,
  `s = sigmoid((log u - log(1-u) + log_alpha) / beta)`,
  `g = clamp(s * (zeta - gamma) + gamma, 0, 1)`.
- Eval-time deterministic: `g_det = clamp(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)`;
  variable *live* iff `g_det > 0.5`.
- Penalty: `L_gate = sum_i sigmoid(log_alpha_i - beta * log(-gamma/zeta))`
  (= expected number of open gates). Loss adds `lambda_gate * L_gate`.

Unlike the width clamp that caused Night-2's gradient death, the hard-concrete
penalty has nonzero gradient whenever a gate is not fully saturated, and the
stretch `(gamma, zeta)` lets gates reach *exactly* 0/1 while log_alpha keeps
receiving gradient through the sample distribution.

## Coupling (both sides must see the same gate)

- **N-side (subspace)**: effective width `w_eff_i = g_i * w_i` is used to build
  interchange masks. A closed gate removes the variable's subspace from every
  swap — the low-level intervention becomes a no-op for that variable.
- **H-side (causal model)**: the discretized value is masked,
  `v_used_i = hard(g_i) * v_i` with straight-through gradient, so a gated-off
  variable is constant-0 in H: swapping it never changes the counterfactual
  label, and the decoder cannot read interventional information from it.

This symmetry is the invariant that keeps IIA honest: a dead variable is a
no-op in *both* N and H, so swaps of dead variables are trivially consistent
and must not be the only thing evaluated (see metrics).

## Metrics

- `gated_k`: parameter-based live count (`g_det > 0.5`).
- `iia_1_live`, `iia_2_live`: IIA with swap sampling restricted to live
  variables — the headline numbers for gated runs, since all-variable IIA is
  inflated by no-op dead swaps. Standard `iia_1`/`iia_2` still reported for
  comparability with Night-1/2 numbers.
- `effective_k` (existing, behavioral flip-rate) is kept as an independent
  check: `gated_k` and `effective_k` should agree at convergence.
- Recovery computed over live variables only.

## Files

- `src/jdas/gates.py` — `VariableGates` module (sample/deterministic/penalty).
- `src/jdas/rotation.py` — `SubspaceLayout` accepts optional gates; gate-scaled
  effective widths feed mask construction.
- `src/jdas/causal_model.py` — learned models accept optional gates; value
  masking with straight-through hard gate.
- `src/jdas/training.py` — `JointConfig.lambda_gate` (default 0.0) +
  `use_gates`; L_gate term; gates included in checkpoints; gate stats in
  history.
- `src/jdas/eval.py` — live-restricted IIA variant; `gated_k`.
- `experiments/run_phase_a.py`, `experiments/run_phase_b.py` — `--gates`,
  `--lambda-gate`, gate metrics in result JSON.

## Implementation notes (deviations from the sketch above)

- **Coupling site.** The single-sample-per-step invariant is enforced in
  `JointTrainer._compute_losses`: it calls `gates.sample(generator=...)` exactly
  once and threads that tensor into `interchange(gate=...)` (N-side widths),
  `H.counterfactual_predict(gate=...)`, `H.predict(gate=...)` (H-side value
  mask), and `layout.total_aligned_dims(gate=...)` (sparsity over gated dims).
  Modules never sample gates themselves.
- **`SubspaceLayout` does not own the gates.** Gate-awareness is an optional
  `gate=` argument on `widths/boundaries/soft_masks/hard_masks/
  total_aligned_dims/hard_widths`, keeping the coupling explicit at the trainer
  rather than hidden behind a stored reference. Closed-gate rows are zeroed both
  by the collapsed (gate-scaled) boundaries and, belt-and-braces, by an explicit
  per-row scale (soft) / liveness mask (hard).
- **Fixed models reject gates.** `FixedCausalModel.{predict,counterfactual_predict}`
  raise on a non-`None` gate (fixed-H baselines keep their exact hypothesis).
  `iia_live` therefore passes the gate to the N-side always but to H only when H
  is a `LearnedCausalModel`; swaps are restricted to live variables regardless.
- **Freeze-and-refit (not in the sketch).** `refit_rotation(gates=...)` bakes the
  discovered liveness into the frozen H — a dead variable's discretized value is
  forced to constant 0 — and then runs plain DAS with `use_gates=False`
  (`lambda_gate=0`), since gates are a training-only mechanism and the refit H is
  already discretized.
- **Checkpoints.** `save_checkpoint(gates=...)` records gate meta + `log_alpha`;
  `load_checkpoint(expect_gates=True|False|None)` raises `CheckpointError` on a
  gate/no-gate mismatch (never a silent default). Pre-gate checkpoints load as
  gate-less.

## Invariants and constraints

- Gates apply only to learned methods (`joint`, `random_rotation`).
- `--gates --lambda-gate 0` is the parameterization control: gates present but
  costless, expected to keep ~all variables live (shows pruning comes from the
  penalty, not the parameterization).
- Gate params are part of the checkpoint; loading a checkpoint without gates
  into a gated config (or vice versa) is an error, not a silent default.
- Training swap sampling stays over all `k_max` variables (dead-swap batches
  are harmless no-ops); only *evaluation* gets the live-restricted variant.

## Experiments (planned)

- Toy: hierarchical_equality layers 1–2 and boolean_comp layer 1, `k_max=4`,
  `lambda_gate ∈ {0, 0.01, 0.03, 0.1, 0.3}`, seeds 0–2. Success: `gated_k=2`,
  live-IIA ≥ 0.9, live pair is a valid basis per the hypothesis library.
- LM: price tagging, Qwen2.5-1.5B l17, capped recipe
  (`--max-width 128 --init-width 32 --sparse-mode per_dim --lambda-sparse 0.02`)
  + gates, `lambda_gate` sweep, seeds 0–1. Success: prune 4→2 at IIA
  comparable to capped das_true (0.891/0.922).
