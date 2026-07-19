# phase_a summary

Mean±std over seeds (n in last column). `-` = metric not applicable.

| task | method | layer | iia_1 | iia_2 | eff_k | recovery | refit_iia_1 | refit_iia_2 | n |
|---|---|---|---|---|---|---|---|---|---|
| boolean_comp | das_true | 0 | 1.000±0.000 | 1.000±0.000 | 2.000±0.000 | - | - | - | 3 |
| boolean_comp | das_true | 1 | 0.845±0.039 | 0.857±0.016 | 2.000±0.000 | - | - | - | 3 |
| boolean_comp | das_true | 2 | 0.797±0.050 | 0.786±0.031 | 2.000±0.000 | - | - | - | 3 |
| boolean_comp | das_wrong | 1 | 1.000±0.000 | 0.000±0.000 | 1.000±0.000 | - | - | - | 3 |
| boolean_comp | joint | 0 | 0.974±0.020 | 0.944±0.032 | 1.667±0.577 | 0.824±0.006 | 0.980±0.012 | 0.951±0.026 | 3 |
| boolean_comp | joint | 1 | 0.975±0.029 | 0.928±0.027 | 1.333±0.577 | 0.800±0.067 | 0.983±0.026 | 0.943±0.002 | 3 |
| boolean_comp | joint | 2 | 0.961±0.024 | 0.931±0.052 | 1.333±0.577 | 0.793±0.037 | 0.965±0.020 | 0.941±0.041 | 3 |
| boolean_comp | random_rotation | 1 | 1.000±0.000 | 0.878±0.006 | 0.000±0.000 | 0.783±0.038 | - | - | 3 |
| hierarchical_equality | das_true | 0 | 0.938±0.024 | 0.945±0.022 | 2.000±0.000 | - | - | - | 3 |
| hierarchical_equality | das_true | 1 | 0.568±0.030 | 0.590±0.014 | 2.000±0.000 | - | - | - | 3 |
| hierarchical_equality | das_true | 2 | 0.505±0.049 | 0.499±0.002 | 2.000±0.000 | - | - | - | 3 |
| hierarchical_equality | das_wrong | 1 | 1.000±0.000 | 0.000±0.000 | 1.000±0.000 | - | - | - | 3 |
| hierarchical_equality | joint | 0 | 0.788±0.192 | 0.779±0.033 | 3.000±1.000 | 0.791±0.201 | 0.794±0.204 | 0.749±0.033 | 3 |
| hierarchical_equality | joint | 1 | 0.964±0.012 | 0.953±0.036 | 3.000±1.000 | 0.908±0.132 | 0.965±0.010 | 0.960±0.031 | 3 |
| hierarchical_equality | joint | 2 | 0.951±0.066 | 0.934±0.102 | 2.000±0.000 | 0.793±0.043 | 0.949±0.068 | 0.932±0.097 | 3 |
| hierarchical_equality | random_rotation | 1 | 0.845±0.059 | 0.747±0.076 | 0.000±0.000 | 0.810±0.112 | - | - | 3 |
