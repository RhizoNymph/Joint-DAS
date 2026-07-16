# Phase A analysis: joint DAS on toy models with known ground truth

Scope: 48 Phase A runs = 2 tasks (`hierarchical_equality`, `boolean_comp`) x 4
methods (`joint`, `das_true`, `das_wrong`, `random_rotation`) x site layers x 3
seeds. The frozen network N is a 3-hidden-layer width-256 ReLU MLP trained to
>99% task accuracy; the intervention site is the post-ReLU activation of a hidden
block (layers 0/1/2 swept for `joint`/`das_true`, layer 1 only for
`das_wrong`/`random_rotation`). Metrics: interchange-intervention accuracy at swap
size |I|=1 (`iia_1`) and |I|=2 with distinct sources (`iia_2`), effective_k
(live variables), and — for learned-model runs — GT recovery score and the
freeze-and-refit IIA. Steps=4000, k_max=4, v=2.

## Summary table (mean +/- std over 3 seeds)

| task | method | layer | iia_1 | iia_2 | eff_k | recovery | refit_iia_1 | refit_iia_2 |
|---|---|---|---|---|---|---|---|---|
| boolean_comp | das_true | 0 | 1.000±0.000 | 1.000±0.000 | 2.00±0.00 | - | - | - |
| boolean_comp | das_true | 1 | 0.845±0.039 | 0.857±0.016 | 2.00±0.00 | - | - | - |
| boolean_comp | das_true | 2 | 0.797±0.050 | 0.786±0.031 | 2.00±0.00 | - | - | - |
| boolean_comp | das_wrong | 1 | 1.000±0.000 | 0.000±0.000 | 1.00±0.00 | - | - | - |
| boolean_comp | joint | 0 | 0.974±0.020 | 0.944±0.032 | 1.67±0.58 | 0.824±0.006 | 0.980±0.012 | 0.951±0.026 |
| boolean_comp | joint | 1 | 0.975±0.029 | 0.928±0.027 | 1.33±0.58 | 0.800±0.067 | 0.983±0.026 | 0.943±0.002 |
| boolean_comp | joint | 2 | 0.961±0.024 | 0.931±0.052 | 1.33±0.58 | 0.793±0.037 | 0.965±0.020 | 0.941±0.041 |
| boolean_comp | random_rotation | 1 | 1.000±0.000 | 0.878±0.006 | 0.00±0.00 | 0.783±0.038 | - | - |
| hierarchical_equality | das_true | 0 | 0.938±0.024 | 0.945±0.022 | 2.00±0.00 | - | - | - |
| hierarchical_equality | das_true | 1 | 0.568±0.030 | 0.590±0.014 | 2.00±0.00 | - | - | - |
| hierarchical_equality | das_true | 2 | 0.505±0.049 | 0.499±0.002 | 2.00±0.00 | - | - | - |
| hierarchical_equality | das_wrong | 1 | 1.000±0.000 | 0.000±0.000 | 1.00±0.00 | - | - | - |
| hierarchical_equality | joint | 0 | 0.788±0.192 | 0.779±0.033 | 3.00±1.00 | 0.791±0.201 | 0.794±0.204 | 0.749±0.033 |
| hierarchical_equality | joint | 1 | 0.964±0.012 | 0.953±0.036 | 3.00±1.00 | 0.908±0.132 | 0.965±0.010 | 0.960±0.031 |
| hierarchical_equality | joint | 2 | 0.951±0.066 | 0.934±0.102 | 2.00±0.00 | 0.793±0.043 | 0.949±0.068 | 0.932±0.097 |
| hierarchical_equality | random_rotation | 1 | 0.845±0.059 | 0.747±0.076 | 0.00±0.00 | 0.810±0.112 | - | - |

(Also emitted machine-readably to `experiments/results/phase_a_summary.md`.)

## Key finding 1: the hand-specified GT model is not representable at deep layers, but a learned one is

On `hierarchical_equality`, DAS with the **true hand-specified** H (`das_true`,
GT variables E1=(a==b), E2=(c==d), decoder y=(E1==E2)) works only at the shallow
site and collapses toward chance as the site deepens:

- layer 0: iia_1 = **0.938±0.024** (seeds 0.957 / 0.910 / 0.945)
- layer 1: iia_1 = **0.568±0.030** (seeds 0.602 / 0.547 / 0.555)
- layer 2: iia_1 = **0.505±0.049** (seeds 0.535 / 0.449 / 0.531) — chance is 0.5

iia_2 collapses in lock-step (0.945 -> 0.590 -> 0.499). effective_k stays 2 (both
GT variables have live masks), so the collapse is not a dead-variable artifact:
no orthogonal rotation into two disjoint subspaces can make N's layer-1/2
counterfactual behavior agree with the literal (E1, E2) factorization.

The **jointly learned** model does not collapse at those same deep sites:

- layer 1: iia_1 = **0.964±0.012**, iia_2 = 0.953±0.036, refit_iia_1 = 0.965±0.010
- layer 2: iia_1 = **0.951±0.066**, iia_2 = 0.934±0.102, refit_iia_1 = 0.949±0.068

with GT recovery scores 0.908±0.132 (L1) and 0.793±0.043 (L2). The
freeze-and-refit IIA tracks the soft-training IIA closely (e.g. L1
0.965 refit vs 0.964 soft), so the discovered H is a genuine hard-discretized
solution, not soft-training slack. Interpretation: joint DAS finds a causal
factorization that IS linearly representable in disjoint subspaces at deep sites
where the proposed hand-specified one is not.

Note this collapse is task/geometry specific: on `boolean_comp`, `das_true`
degrades much more gently with depth (L0 1.000 -> L1 0.845 -> L2 0.797), i.e. the
GT (x1&x2, x3) factorization stays largely representable through the MLP. The
dramatic layer-1/2 failure is specific to hierarchical equality.

## Key finding 2: the "wrong" single-output-copy H — and an honest caveat about its iia_2

`das_wrong` uses a deliberately degenerate H: a single variable Z = y (the label
itself), k=1. It achieves **iia_1 = 1.000** on both tasks (a single output-copy
variable trivially reproduces any single-source counterfactual, since the
counterfactual label just equals the swapped source's label). Its reported
**iia_2 = 0.000**.

Honest caveat — this 0.0 is a **reporting artifact, not a real |I|=2
evaluation.** With k=1, there is no second variable to co-swap. `eval.iia` filters
`swap_sizes` to those `<= layout.k_max`, so swap size 2 is skipped entirely and
the returned dict is `{1: 1.0}`. The trainer's `_evaluate` then does
`iia_scores.get(2, 0.0)`, materializing a **default 0.0** for a metric that was
never computed. So das_wrong's iia_2 should be read as "not applicable / no
composed test exists for k=1," not "scored 0 on composed swaps." The intended
falsification of an output-copy H (that it cannot reproduce multi-source
composed counterfactuals) is therefore **not** demonstrated by this number; it
would require a k>=2 wrong H with one output-copy variable plus a second live
variable. This is a limitation of the current das_wrong baseline design, and the
0.0 in the table should be footnoted rather than cited as evidence.

## Key finding 3: the random-rotation control shows H can be vacuously satisfied

`random_rotation` (learned H, but Q frozen at random init) gets high iia_1
(hierarchical 0.845±0.059; boolean 1.000±0.000) yet **effective_k = 0** in every
run: no single-variable subspace swap flips N's output by more than the 2%
liveness threshold. The agreement is vacuous — H's encoders/decoder drift to a
degenerate solution that "predicts" the intervention outcome without the rotation
carrying any causal content (its recovery score, 0.78-0.81, is also just the
base-rate agreement a constant/relabeled variable achieves, not real recovery).
This is the intended control result: high IIA alone is not evidence of a found
causal representation; effective_k and recovery are needed to rule out
degeneracy. It also shows Q is doing real work in the `joint` runs (which reach
effective_k 2-3).

## Introspection (hierarchical_equality, joint, layer 1, seed 0)

A retrained joint model (`experiments/introspect_phase_a.py`, same code path as
`run_phase_a.py`, 4000 steps, CPU) reproduces the seed-0 layer-1 run: final
iia_1 = 0.977, iia_2 = 0.980, effective_k = 2, recovery_score = 0.797,
refit_iia_1 = 0.973, refit_iia_2 = 0.988, hard widths [31, 30, 31, 30]. Full
artifacts in `experiments/results/introspect_hier_l1_s0.{json,md}`. (The retrain
was on CPU and its recovery_score 0.797 differs slightly from the committed GPU
run's 0.756 due to RNG/device; both are far below clean GT recovery and both
correspond to the same alternative-basis solution characterized below.)

**Variable-hypothesis agreement** (best value relabeling, 4096 fresh inputs;
effect_rate = fraction of inputs where a single-variable swap flips N's output):

| var | width | effect_rate | E1 | E2 | XOR(E1,E2) | label y | const0 |
|---|---|---|---|---|---|---|---|
| Z0 | 31 | 0.000 | 0.848 | 0.563 | 0.562 | 0.562 | 0.660 |
| Z1 | 30 | 0.001 | 0.675 | 0.588 | 0.521 | 0.521 | 0.556 |
| Z2 | 31 | **0.309** | 0.739 | 0.743 | 0.751 | 0.751 | 0.745 |
| Z3 | 30 | **0.317** | 0.747 | 0.745 | 0.746 | 0.746 | 0.762 |

Only Z2 and Z3 are causally live (effect ~0.31; Z0/Z1 have effect ~0, dead
despite wide masks). Neither live variable matches any single hypothesis strongly
(all agreements ~0.74-0.75): the layer-1 learned variables are **NOT** the literal
GT atoms {E1, E2}.

**What they are.** The joint (Z3, Z2) vs (E1, E2) table pins it down exactly
(cells are the modal (Z3, Z2) per GT combination, mean purity **0.989**):

| E1 | E2 | (Z3, Z2) | purity |
|---|---|---|---|
| 0 | 0 | (1, 0) | 0.980 |
| 0 | 1 | (1, 1) | 0.990 |
| 1 | 0 | (1, 1) | 0.986 |
| 1 | 1 | (0, 1) | 1.000 |

Reading off the two learned variables as Boolean functions of (E1, E2):

- **Z2 = E1 OR E2** (0 only when both equalities fail),
- **Z3 = NAND(E1, E2) = NOT(E1 AND E2)** (0 only when both equalities hold).

So joint DAS discovered the factorization **(OR(E1,E2), NAND(E1,E2))** rather than
(E1, E2). This is a valid alternative basis: (OR, NAND) jointly determine (E1, E2)
up to the E1<->E2 symmetry (they cannot separate the (0,1) and (1,0) cases, which
is exactly why each variable's marginal agreement with E1 or E2 caps at ~0.75),
and — crucially — the task label y = (E1 == E2) = NOT(OR AND NOT-NAND) is fully
recoverable from (OR, NAND). That is why iia_1/iia_2 reach ~0.98 while marginal
recovery is only ~0.75. The learned decoder confirms it collapses (Z2, Z3) to the
correct label. Joint cell purity 0.989 shows the two live variables jointly and
almost-deterministically encode (E1, E2).

**Conclusion for the paper-level claim:** at layer 1, seed 0, joint DAS does NOT
recover the literal GT atoms; it finds an equally valid *alternative* two-variable
factorization (OR, NAND) of the same information. Combined with seeds 1-2 (which do
recover {E1, E2} cleanly), the honest claim is: joint DAS finds a valid
two-variable causal factorization at deep layers where the hand-specified GT
model fails, but the specific basis it lands on is seed-dependent (sometimes the
GT atoms, sometimes a relabeled/rotated equivalent such as (OR, NAND)).

By contrast, **seeds 1 and 2** at layer 1 recover the GT atoms cleanly
(recovery 0.998 and 0.970). So across seeds the layer-1 joint solution is
sometimes exactly {E1, E2} and sometimes an alternative valid factorization. The
honest paper-level claim is therefore: **joint DAS finds a valid two-variable
factorization at deep layers where the hand-specified GT model fails; whether that
factorization coincides with the literal GT atoms is seed-dependent (2/3 seeds
yes, 1/3 an alternative basis).**

## Training dynamics (representative joint run, hierarchical L1 s0)

iia vs step (eval every 400 steps):

| step | iia_1 | iia_2 | eff_k |
|---|---|---|---|
| 0 | 0.504 | 0.598 | 1 |
| 400 | 0.551 | 0.609 | 4 |
| 800 | 0.684 | 0.730 | 3 |
| 1200 | 0.945 | 0.945 | 2 |
| 1600 | 0.980 | 0.973 | 2 |
| 2400 | 0.988 | 0.992 | 2 |
| 3200 | 0.992 | 0.988 | 2 |
| 3999 | 0.977 | 0.980 | 2 |

The model briefly opens all four variables (eff_k 4 at step 400) as the Gumbel/
mask temperatures are still high, then prunes to the effective 2 by step ~1200,
at which point IIA jumps to >0.94 and stabilizes near 0.98. This "expand then
consolidate" trajectory is typical of the joint runs.

## Limitations

- **Recovery variance across seeds (joint).** hierarchical L0 is the worst case:
  recovery 0.791±0.201 driven by one bad seed (s1: iia_1 0.566, recovery 0.597)
  vs. two good ones (s0 0.777, s2 0.998). L1 recovery is 0.908±0.132 for the same
  reason (seed-0 alternative-basis run drags the mean). Joint training does not
  reliably converge to the *literal* GT atoms every seed, even when it reliably
  achieves high IIA. IIA is far more stable than recovery.
- **Wide masks / weak sparsity pressure.** With lambda_sparse=0.1 the learned
  blocks stay large: hard widths are consistently ~31 dims each of d=256 (e.g.
  [31,31,30,30]), i.e. the effective variables occupy ~1/8 of the site each and a
  large fraction of the 256 dims remains aligned. The sparsity term is too weak to
  drive minimal subspaces; the found variables are correct in behavior but not
  dimension-minimal. Reported effective_k relies on the liveness (flip-rate)
  test, not on masks actually shrinking to a few dims.
- **iia_2-for-k=1 caveat (see Finding 2).** The das_wrong iia_2 = 0.0 is a
  skipped-metric default, not a computed composed-swap failure. Do not cite it as
  falsifying the output-copy hypothesis.
- **boolean_comp k_eff = 1 / coarser abstraction.** Several boolean_comp joint
  runs settle at effective_k = 1 (L0 s0; L1 s1, s2; L2 s1, s2) while still
  reaching iia_1 ~0.97-0.99 and iia_2 ~0.93-0.97. Because y = (x1&x2) | x3, a
  single well-chosen variable (e.g. tracking the label y, or the OR itself)
  already reproduces most single- and even two-source counterfactuals on this
  task, so the model can pass IIA with a coarser one-variable abstraction rather
  than the intended two-variable (x1&x2, x3) structure. Recovery scores there
  (~0.75-0.83) are correspondingly modest. This is a genuine identifiability
  limitation of boolean_comp as a recovery benchmark, not a training failure:
  the multi-source pressure is weaker when the task's composed counterfactuals
  are largely determined by one aggregate variable.
- **das_true depth-collapse is task-specific.** The clean layer-1/2 collapse is a
  hierarchical-equality phenomenon; boolean_comp das_true degrades only mildly.
  The "GT not representable at depth" claim is demonstrated for hierarchical
  equality, not as a universal property.

## What is and isn't shown

Shown: (i) on hierarchical equality the hand-specified GT alignment fails at
deep sites (-> chance) while jointly learned H stays ~0.95-0.96 IIA with valid
freeze-and-refit; (ii) the random-rotation control confirms high IIA can be
vacuous (effective_k 0) and that Q does real work in joint runs; (iii) joint DAS
recovers a valid two-variable factorization at deep layers, coinciding with the
literal GT atoms in 2/3 layer-1 seeds and, in the third (seed 0), the alternative
but provably-valid basis (Z2, Z3) = (OR(E1,E2), NAND(E1,E2)) with joint cell
purity 0.989 — confirmed by direct introspection of the retrained model.

Not shown: (i) reliable *literal* GT-atom recovery every seed; (ii) dimension-
minimal subspaces (masks stay wide); (iii) a proper composed-swap falsification of
the output-copy hypothesis (blocked by the k=1 das_wrong design); (iv) that the
das_true depth-collapse generalizes beyond hierarchical equality.
