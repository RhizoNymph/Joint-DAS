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

## Gate optimization

Gates need their **own learning rate** and a **configurable init**, because the
gate parameters live on a very different scale from `Q`/boundaries/`H`.

- **Adam speed limit (the failure mode).** Adam's update is bounded by roughly
  `±lr` per step (the moment ratio `m/√v` saturates near ±1). With the default
  `lr=1e-3` and 800–1500 steps, `log_alpha` (init `+2.0`) can move at most
  ~`lr*steps`, nowhere near the `≈ -0.4` threshold where `g_det` crosses 0.5
  and a gate goes dead. In the LM runs this made `lambda_gate` inert:
  `g_det`/`gated_k` trajectories were near-identical for `lambda_gate ∈ {0,
  0.2}` — the penalty had a gradient, but the optimizer could not act on it fast
  enough. This is *not* gradient death (contrast the Night-2 width clamp); it is
  an optimizer-step-size limit.
- **Two knobs** (`JointConfig`, both threaded from `--gate-lr`/`--gate-init` in
  `run_phase_a.py`/`run_phase_b.py` and recorded in the result JSON + checkpoint
  meta):
  - `gate_lr: float | None = None` — a dedicated learning rate for the gate
    param group (`None` falls back to `config.lr`). The trainer puts gate
    params in their own optimizer group so this lr applies only to them. A value
    like `0.05` lets a gate travel `+2.0 → -0.4` in a few tens of steps, so the
    L0 penalty can actually close gates.
  - `gate_init: float = 2.0` — the initial `log_alpha`. Lower it (or set it
    negative, e.g. `-1.0`) to start gates near/below the live threshold when you
    want closing to be the default rather than the exception.
- **Checkpoints.** `gate_init`/`gate_lr` are stored in the checkpoint gate meta;
  pre-existing gated checkpoints without these keys still load (they default to
  `2.0`/`None`). `log_alpha` itself is restored from the state dict, so
  `gate_init` is informational on reload.

### Bistability and the training schedule (RESULTS.md N3.3)

With `gate_lr=0.05` (matched to the run horizon) the L0 term is *still* not the
deciding force — the gate system is **bistable**, and whichever gradient
dominates *while gates are still mobile* wins the race:

- **Toy → all-open (penalty saturation).** CF-usefulness gradients push every
  `log_alpha` up early; once a gate is confidently open the penalty derivative
  `sigmoid'(log_alpha − β·log(−γ/ζ))` is exponentially small, so no `λ ≤ 0.3`
  can pull a saturated-open gate back within the horizon. (The project's third
  λ-independent gradient-death instance, after the N2.1 width clamp and the
  N3.1 gate-optimizer speed limit.)
- **LM → all-closed (CF collapse).** The CF task is hard early at LM scale, so
  the easiest descent direction for a fast gate parameter is to *close*
  everything: a closed gate makes every interchange a no-op and the
  counterfactual target collapses to the clean label. Gates die by step ~80
  even at `λ_gate = 0`, proving the collapse is the CF gradient, not the penalty.

The fix is not a bigger `λ` — it is controlling the schedule so variables
become causally useful *first*, then applying pruning pressure gently, and
keeping both saturation regions gradient-alive. Three `JointConfig` knobs (all
threaded from `--gate-warmup`/`--gate-lambda-ramp`/`--gate-clamp` in
`run_phase_a.py`/`run_phase_b.py`, recorded in checkpoint gate meta, and echoed
per-eval in history as `gate_phase` + `lambda_gate_eff`):

- `gate_warmup_steps: int = 0` — while `step < gate_warmup_steps`, training is
  **numerically identical to a no-gates run**: `gate=None` is threaded
  everywhere (full widths, no H-side value mask), the gate penalty is omitted
  from the loss, and the gate params receive no updates (the forward never
  touches `log_alpha`, so its grad stays `None` and AdamW skips it). This lets H
  distribute the computation across the variables before any gate can close.
- `gate_lambda_ramp_steps: int = 0` — after warmup, the effective `λ_gate`
  scales linearly `0 → config.lambda_gate` over this many steps (`0` = instant
  full λ). Gentle onset avoids the sudden penalty spike that snaps mobile gates
  into the closed basin.
- `gate_clamp: float | None = 3.0` — after each *active* optimizer step,
  `log_alpha` is clamped in place to `[−gate_clamp, +gate_clamp]` so neither the
  open- nor closed-saturation region kills the penalty/sample gradient (`None`
  disables). The clamp is a no-op during warmup, preserving the warmup ==
  no-gates invariant.

The **gate phase** at a step is `warmup` (`step < gate_warmup_steps`), `ramp`
(within the ramp window), or `active` (full λ); it is recorded in every history
record for gated runs so the dynamics are visible in the result JSONs.

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
  `use_gates`; `gate_lr`/`gate_init` optimization knobs (separate optimizer
  param group for gates); `gate_warmup_steps`/`gate_lambda_ramp_steps`/
  `gate_clamp` schedule knobs (`_gates_active`/`_gate_phase`/
  `_effective_lambda_gate` helpers + `_post_step` clamp); L_gate term; gates
  included in checkpoints; gate stats + `gate_phase`/`lambda_gate_eff` in history.
- `src/jdas/eval.py` — live-restricted IIA variant; `gated_k`.
- `experiments/run_phase_a.py`, `experiments/run_phase_b.py` — `--gates`,
  `--lambda-gate`, `--gate-lr`, `--gate-init`, gate metrics in result JSON.

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
