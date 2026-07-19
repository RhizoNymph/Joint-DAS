# Discrete search baseline: hierarchical_equality (layer 1)

Config: seed=0 steps=1500 device=cuda candidates=['E1', 'E2', 'XNOR(=y)', 'AND', 'OR', 'NAND']

Ranking by combined score (mean of iia_1, iia_2), best first.

| rank | V1 | V2 | clean_task_acc | iia_1 | iia_2 | combined |
|---|---|---|---|---|---|---|
| 1 | E1 | XNOR(=y) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | E2 | XNOR(=y) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | AND | OR | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | OR | NAND | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | XNOR(=y) | OR | 1.0000 | 0.9863 | 0.9766 | 0.9814 |
| 6 | E2 | NAND | 0.7624 | 0.8926 | 0.9336 | 0.9131 |
| 7 | E2 | AND | 0.7624 | 0.8926 | 0.9297 | 0.9111 |
| 8 | XNOR(=y) | NAND | 1.0000 | 0.8926 | 0.8594 | 0.8760 |
| 9 | XNOR(=y) | AND | 1.0000 | 0.8906 | 0.8516 | 0.8711 |
| 10 | E1 | OR | 0.7624 | 0.8496 | 0.8887 | 0.8691 |
| 11 | E2 | OR | 0.7466 | 0.7949 | 0.8457 | 0.8203 |
| 12 | E1 | NAND | 0.7468 | 0.7832 | 0.8281 | 0.8057 |
| 13 | E1 | AND | 0.7468 | 0.7812 | 0.8281 | 0.8047 |
| 14 | AND | NAND | 0.7468 | 0.6641 | 0.7129 | 0.6885 |
| 15 | E1 | E2 | 1.0000 | 0.5586 | 0.5996 | 0.5791 |

Best pair: **(E1, XNOR(=y))** with combined 1.0000 (iia_1=1.0000).
