# Introspection: hier_l1_s0

Config: task=`hierarchical_equality` method=joint site_layer=1 seed=0 steps=4000 k_max=4 v=2 device=cpu

Final IIA: iia_1=0.9766, iia_2=0.9805, effective_k=2, recovery_score=0.7969, refit_iia_1=0.9727, refit_iia_2=0.9883

Hard mask widths: [31, 30, 31, 30]


## Variable-hypothesis agreement (best value relabeling)

| Variable | width | effect_rate | GT0_E1 | GT1_E2 | XOR(E1,E2) | label_y | const0 |
|---|---|---|---|---|---|---|---|
| Z0 | 31 | 0.000 | 0.848 | 0.563 | 0.562 | 0.562 | 0.660 |
| Z1 | 30 | 0.001 | 0.675 | 0.588 | 0.521 | 0.521 | 0.556 |
| Z2 | 31 | 0.309 | 0.739 | 0.743 | 0.751 | 0.751 | 0.745 |
| Z3 | 30 | 0.317 | 0.747 | 0.745 | 0.746 | 0.746 | 0.762 |

`effect_rate` = fraction of inputs where a single-variable swap of that variable flips N's output (liveness).


## Per-variable best match

- **Z0** (width 31, effect 0.000): best = `GT0_E1` @ 0.848
- **Z1** (width 30, effect 0.001): best = `GT0_E1` @ 0.675
- **Z2** (width 31, effect 0.309): best = `XOR(E1,E2)` @ 0.751
- **Z3** (width 30, effect 0.317): best = `const0` @ 0.762

## Learned decoder truth table (effective variables)

Effective (live) variable indices: [2, 3]
Dead variables held at default values: [0, 0, 1, 1]

| Z2 | Z3 | pred_label | logits |
|---|---|---|---|
| 0 | 0 | 1 | [-8.4044, 8.5241] |
| 0 | 1 | 1 | [-2.982, 2.9365] |
| 1 | 0 | 1 | [-4.1388, 4.019] |
| 1 | 1 | 0 | [2.9425, -3.0009] |

## Joint (Z3, Z2) vs (E1, E2) [two highest-effect variables]

| E1 | E2 | n | modal (Zi,Zj) | purity |
|---|---|---|---|---|
| 0 | 0 | 1041 | [1, 0] | 0.980 |
| 0 | 1 | 1044 | [1, 1] | 0.990 |
| 1 | 0 | 1037 | [1, 1] | 0.986 |
| 1 | 1 | 974 | [0, 1] | 1.000 |

## Interpretation

The effective learned variables do NOT both map cleanly onto {E1, E2}. Per-variable best matches among effective vars: Z2->XOR(E1,E2) (0.751), Z3->const0 (0.762). This suggests an alternative (possibly relabeled/rotated) but valid factorization rather than the literal GT atoms. Joint (Z3,Z2) vs (E1,E2): mean cell purity 0.989 (1.0 => the top-two variables jointly determine (E1,E2)).
