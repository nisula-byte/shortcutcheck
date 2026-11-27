# Exploratory: confidence drop with sign reversed (not pre-registered)

confidence_reversed = minus the pre-specified confidence drop, fixed before computing. Spearman rho with F1 decay to 2015-16; 95% cluster-bootstrap CIs (2000 resamples, seed 11); adj. = after removing algorithm mean ranks.

| Dataset | Signal | n | rho | 95% CI | adj. rho | 95% CI |
|---|---|---|---|---|---|---|
| AndroCT 2015-16 | mr4_monitor | 150 | 0.852 | [0.751, 0.907] | 0.579 | [0.322, 0.767] |
| AndroCT 2015-16 | psi | 150 | 0.196 | [-0.177, 0.547] | 0.151 | [-0.060, 0.328] |
| AndroCT 2015-16 | confidence_reversed | 150 | 0.668 | [0.521, 0.740] | -0.210 | [-0.402, -0.026] |
| AZL 2015-16 | mr4_monitor | 150 | 0.479 | [0.109, 0.750] | 0.564 | [0.279, 0.748] |
| AZL 2015-16 | psi | 150 | 0.652 | [0.445, 0.757] | 0.357 | [-0.003, 0.610] |
| AZL 2015-16 | confidence_reversed | 150 | 0.866 | [0.757, 0.916] | 0.366 | [-0.222, 0.663] |

Script commit b66a66d35de312c9d432cafa28535e396fce0971; MLflow run b6cbcaee42c34e74842a804141db54f1.
