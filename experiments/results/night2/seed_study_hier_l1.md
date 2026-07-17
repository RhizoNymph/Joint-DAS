# Seed / basis variance study: hierarchical_equality (layer 1)

Config: seeds=10 steps=4000 k_max=4 v=2 device=cuda

## Per-seed classification

| seed | class | live vars (fn) | iia_1 | iia_2 | eff_k | recovery |
|---|---|---|---|---|---|---|
| 0 | other | Z0=notB, Z1=A, Z2=AND | 0.9844 | 0.9727 | 3 | 0.9941 |
| 1 | other | Z1=OR, Z2=NAND | 0.9883 | 0.9922 | 2 | 0.7549 |
| 2 | other | Z0=A, Z1=B, Z2=AND, Z3=OR | 0.9531 | 0.9727 | 4 | 0.9951 |
| 3 | other | Z0=NOR, Z1=OR, Z2=AND | 0.9375 | 0.9336 | 3 | 0.7676 |
| 4 | other | Z0=A, Z1=notA, Z2=notA, Z3=B | 0.8711 | 0.7969 | 4 | 1.0000 |
| 5 | other | Z0=notA, Z1=B, Z3=XOR | 0.9609 | 0.9297 | 3 | 0.9736 |
| 6 | other | Z0=notA, Z1=NAND, Z2=notB, Z3=A | 1.0000 | 0.9844 | 4 | 0.9951 |
| 7 | other | Z0=NOR, Z3=AND | 0.9727 | 0.9844 | 2 | 0.8564 |
| 8 | other | Z2=NAND, Z3=OR | 0.9805 | 0.9609 | 3 | 0.7803 |
| 9 | other | Z1=OR, Z2=AND | 0.9844 | 0.9688 | 2 | 0.7490 |

## Aggregate

Class counts:
- `atoms`: 0
- `equivalent_basis`: 0
- `output_copy`: 0
- `other`: 10

IIA: iia_1 = 0.9633 ± 0.0354, iia_2 = 0.9496 ± 0.0546 (over 10 seeds).
