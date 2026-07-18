# Night-3 gate sweeps

## Toy sweeps

### hierarchical_equality (layer 1)  —  5 runs

| lambda_gate | seed | gated_k | effective_k | iia_1_live | iia_2_live | iia_1 | iia_2 | recovery_score | aligned_dims_gated | prune_step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.030 | 0 | 4 | 2.00 | 0.934 | 0.930 | 0.906 | 0.918 | 0.725 | 126.00 | 0 |
| 0.030 | 1 | 4 | 2.00 | 0.965 | 0.957 | 0.934 | 0.941 | 0.814 | 126.00 | 0 |
| 0.030 | 2 | 4 | 2.00 | 0.535 | 0.559 | 0.570 | 0.543 | 0.671 | 126.00 | 0 |
| 0.100 | 0 | 4 | 2.00 | 0.961 | 0.953 | 0.953 | 0.918 | 0.741 | 126.00 | 0 |
| 0.100 | 1 | 4 | 3.00 | 0.895 | 0.914 | 0.910 | 0.918 | 0.762 | 127.00 | 0 |

_Per-lambda aggregate (mean over seeds):_

| lambda_gate | n_seeds | mean gated_k | mean iia_1_live | mean iia_2_live |
|---|---|---|---|---|
| 0.030 | 3 | 4.00 | 0.811 | 0.815 |
| 0.100 | 2 | 4.00 | 0.928 | 0.934 |


## LM sweep (Qwen2.5-1.5B, layer 17, capped)

_Night-2 capped anchors (reference):_

| method | iia_1 | iia_2 | k_eff | aligned dims |
|---|---|---|---|---|
| capped joint | 0.855 | 0.863 | 4 | 59 |
| capped das_true | 0.891 | 0.922 | 4 | 32 |
| capped random_rotation | 0.781 | 0.730 | - | - |

### gates ? (layer 17)  —  7 runs

| lambda_gate | seed | gated_k | effective_k | iia_1_live | iia_2_live | iia_1 | iia_2 | recovery_score | aligned_dims_gated | prune_step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0 | 4 | 4.00 | 0.707 | 0.793 | 0.801 | 0.816 | 0.822 | 71.00 | 0 |
| 0.010 | 0 | 4 | 4.00 | 0.723 | 0.777 | 0.797 | 0.816 | 0.821 | 70.00 | 0 |
| 0.010 | 1 | 4 | 3.00 | 0.848 | 0.840 | 0.812 | 0.836 | 0.730 | 68.00 | 0 |
| 0.050 | 0 | 4 | 4.00 | 0.742 | 0.820 | 0.836 | 0.855 | 0.829 | 72.00 | 0 |
| 0.050 | 1 | 4 | 3.00 | 0.875 | 0.855 | 0.816 | 0.820 | 0.717 | 68.00 | 0 |
| 0.200 | 0 | 4 | 4.00 | 0.719 | 0.801 | 0.789 | 0.832 | 0.828 | 71.00 | 0 |
| 0.200 | 1 | 4 | 3.00 | 0.863 | 0.812 | 0.805 | 0.793 | 0.704 | 67.00 | 0 |

_Per-lambda aggregate (mean over seeds):_

| lambda_gate | n_seeds | mean gated_k | mean iia_1_live | mean iia_2_live |
|---|---|---|---|---|
| 0.000 | 1 | 4.00 | 0.707 | 0.793 |
| 0.010 | 2 | 4.00 | 0.785 | 0.809 |
| 0.050 | 2 | 4.00 | 0.809 | 0.838 |
| 0.200 | 2 | 4.00 | 0.791 | 0.807 |

