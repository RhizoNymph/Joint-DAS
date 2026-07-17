# Night 2 summary — Joint-DAS

Compact tables for the Night-2 experiments. All numbers read directly from the
result JSONs under `experiments/results/night2/` and `experiments/results/phase_b/`.
Fits: Qwen2.5-1.5B-Instruct (LM) / 3-layer width-256 MLP (toy), seed 0 unless noted.

## 1. Collapse mechanism (uncapped, normalized penalty)

`pt_joint_l17_s0_sparse{50,200}.json` are **identical in every training-trajectory
field** (recovery, per-step iia/k_eff/aligned_dims/hard_widths, final). Only
`loss_total` differs, by exactly `(200−50)·loss_sparse = 150·1.0 = 150` at every
step. `aligned_dims` is pinned at 1536 and `loss_sparse` pinned at 1.0 from step 0;
`hard_widths = [1536,0,0,0]` throughout. Raising λ 4× changes nothing because the
penalty `λ·clamp(cumsum(widths), max=d)/d` is saturated at the clamp — gradient
exactly 0 at any λ.

| λ_sparse | final iia_1 | final iia_2 | k_eff | hard_widths | aligned_dims | recovery |
|---|---|---|---|---|---|---|
| 50  | 0.809 | 0.832 | 1 | [1536,0,0,0] | 1536 | 0.724 |
| 200 | 0.809 | 0.832 | 1 | [1536,0,0,0] | 1536 | 0.724 |

## 2. Capped-LM comparison (Qwen2.5-1.5B, layer 17, per_dim penalty, max_width 128, init 32, λ_sparse 0.02, 1200 steps, no refit)

| method (file) | position | iia_1 | iia_2 | k_eff | hard_widths | aligned_dims (/1536) | recovery |
|---|---|---|---|---|---|---|---|
| joint (`pt_joint_l17_capped`) | last | **0.855** | **0.863** | 4 | [15,15,15,14] | 59.3 (3.9%) | 0.719 |
| random control (`pt_random_l17_capped`) | last | 0.781 | 0.730 | **0** | [14,14,13,14] | 55.3 (3.6%) | 0.724 |
| joint z-digits (`pt_joint_l17_zdigits_capped`) | z_digits | 0.789 | 0.730 | 3 | [14,13,14,14] | 54.9 (3.6%) | 0.731 |

Joint − control: **iia_1 +0.074, iia_2 +0.133**, with 4 live variables vs the
control's 0 (vacuous). The cap prevents whole-space collapse, so the control can no
longer cheat, and the learned rotation does demonstrable causal work at LM scale.
Recovery ~0.72 = partial; no clean (L,U) recovery.

## 3. Seed / basis study (hier-eq, layer 1, 10 seeds, 4000 steps)

Aggregate: iia_1 0.963±0.035, iia_2 0.950±0.055. Classifier labels all 10 `other`
(its ≥0.9 joint-purity chain is stricter than the data; per-variable `fn_agreement`
is 0.97–1.00 for the live vars). `atoms`/`equivalent_basis`/`output_copy` counts all 0.

| seed | live vars (fn @ agreement) | iia_1 | iia_2 | eff_k | recovery | basis |
|---|---|---|---|---|---|---|
| 0 | notB .998, A .999, AND 1.00 | 0.984 | 0.973 | 3 | 0.994 | overcomplete (3) |
| 1 | OR .984, NAND 1.00 | 0.988 | 0.992 | 2 | 0.755 | minimal 2-var (OR,NAND) |
| 2 | A .998, B .996, AND .999, OR .870 | 0.953 | 0.973 | 4 | 0.995 | overcomplete (4): A,B,AND,OR |
| 3 | NOR .960, OR .952, AND 1.00 | 0.938 | 0.934 | 3 | 0.768 | overcomplete (3) |
| 4 | A .984, notA .997, notA .710, B .999 | 0.871 | 0.797 | 4 | 1.000 | overcomplete (4): A,notA,notA,B |
| 5 | notA .993, B .748, XOR .753 | 0.961 | 0.930 | 3 | 0.974 | overcomplete (3) |
| 6 | notA .757, NAND .999, notB .997, A .993 | 1.000 | 0.984 | 4 | 0.995 | overcomplete (4) |
| 7 | NOR .980, AND .999 | 0.973 | 0.984 | 2 | 0.856 | minimal 2-var (NOR,AND) |
| 8 | NAND .999, OR .982 | 0.980 | 0.961 | 3ᵃ | 0.780 | minimal 2-var (NAND,OR) |
| 9 | OR .983, AND .999 | 0.984 | 0.969 | 2 | 0.749 | minimal 2-var (OR,AND) |

ᵃ seed 8 has 2 vars in `live_fns` (Z2,Z3) but reports `effective_k = 3` (a third
mask passes the liveness threshold with near-zero effect_rate 0.018); the two named
vars are the causally meaningful ones.

Headline: 10/10 learn causally valid variable sets with near-perfect boolean
semantics; 0/10 recover exactly `{E1,E2}`. 4 seeds (1,7,8,9) find a minimal 2-var
alternative basis; 6 find overcomplete 3–4-var sets. Widths stay ~30/256 (sparsity
too weak at toy scale to prune redundant vars).

## 4. Search baseline (brute-force, fit Q per candidate pair)

### hier-eq (layer 1) — top 5 + E1+E2 (last of 15)

| rank | pair | clean_task_acc | iia_1 | iia_2 |
|---|---|---|---|---|
| 1 | E1 + XNOR(=y) | 1.000 | 1.000 | 1.000 |
| 2 | E2 + XNOR(=y) | 1.000 | 1.000 | 1.000 |
| 3 | AND + OR | 1.000 | 1.000 | 1.000 |
| 4 | OR + NAND | 1.000 | 1.000 | 1.000 |
| 5 | XNOR(=y) + OR | 1.000 | 0.986 | 0.977 |
| … | … | … | … | … |
| **15 (last)** | **E1 + E2** | 1.000 | **0.559** | 0.600 |

The literal atom pair `E1+E2` ranks LAST despite `clean_task_acc = 1.0` (it
reconstructs the label but is not a valid DAS alignment at this site). Two perfect-IIA
pairs (E1+XNOR, E2+XNOR) contain the output variable y=XNOR itself: an
(output + one-atom) pair is also a valid basis — it jointly determines both atoms up
to the E1↔E2 symmetry.

### boolean_comp (layer 1) — top 5 (of 15)

| rank | pair | clean_task_acc | iia_1 | iia_2 |
|---|---|---|---|---|
| 1 | x3 + OR(=y) | 1.000 | 0.945 | 0.938 |
| 2 | A + OR(=y) | 1.000 | 0.918 | 0.955 |
| 3 | OR(=y) + notA | 1.000 | 0.916 | 0.936 |
| 4 | OR(=y) + notx3 | 1.000 | 0.910 | 0.932 |
| 5 | x3 + notx3 | 0.879 | 0.869 | 0.895 |

Best pair `x3+OR(=y)` (0.945/0.938) — a coarser (contains y) solution, consistent
with the joint method's convergence to composite bases.

## 5. Wrong-composition falsification (das_wrong_and, k=2, 3 seeds each; sample std)

`das_wrong_and` uses a k=2 hypothesis with the wrong composition law (AND where the
true law is XNOR for hier / OR for bool). Analytic `agreement_ceiling` (|I|=1) is
the best achievable IIA for that wrong law by construction.

| group | iia_1 (measured) | iia_2 (measured) | analytic ceiling (|I|=1 / |I|=2) | verdict |
|---|---|---|---|---|
| hier L0 | 0.568±0.059 | 0.576±0.010 | 0.747 / 0.750 | at/below ceiling |
| hier L1 | 0.622±0.031 | 0.673±0.023 | 0.747 / 0.750 | at/below ceiling |
| bool L0 | 0.814±0.014 | 0.827±0.014 | 0.876 / 0.875 | at/below ceiling |
| bool L1 | 0.720±0.029 | 0.729±0.008 | 0.876 / 0.875 | at/below ceiling |

Every wrong-AND run sits at or below its analytic ceiling and far below the correct
models (Phase A das_true L0 0.94–1.0; capped joint LM ~0.855). The framework
demonstrably falsifies wrong-but-plausible causal laws under composed interventions —
a real k≥2 falsification, closing Night 1's `das_wrong` (k=1) gap.

## Wave-B status

Not present when this summary was written: `pt_das_true_l17_capped.json`,
`pt_joint_l17_capped_s1.json`, `pt_joint_l10_zdigits_capped.json` — mark as running.
The capped-LM plot/table will gain a `das_true (capped)` bar automatically once the
file appears (the plotter reads it if present).

## Plots

`docs/assets/night2_capped_lm.png`, `night2_seed_basis.png`, `night2_search_hier.png`,
`night2_wrong_and.png`. Regenerate with
`uv run python experiments/analyze_night2.py`.
