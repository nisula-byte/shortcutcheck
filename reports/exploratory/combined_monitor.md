# Exploratory: MR4 monitor combined with PSI (not pre-registered)

Combined = mean of within-table ranks of MR4 and PSI, fixed before computing. Spearman rho with F1 decay to 2015-16; 95% cluster-bootstrap CIs (2000 resamples, seed 11); adj. = after removing algorithm mean ranks.

| Dataset | Signal | n | rho | 95% CI | adj. rho | 95% CI |
|---|---|---|---|---|---|---|
| AndroCT 2015-16 | mr4_monitor | 150 | 0.852 | [0.751, 0.907] | 0.579 | [0.322, 0.767] |
| AndroCT 2015-16 | psi | 150 | 0.196 | [-0.177, 0.547] | 0.151 | [-0.060, 0.328] |
| AndroCT 2015-16 | combined | 150 | 0.715 | [0.550, 0.814] | 0.419 | [0.293, 0.528] |
| AZL 2015-16 | mr4_monitor | 150 | 0.479 | [0.109, 0.750] | 0.564 | [0.279, 0.748] |
| AZL 2015-16 | psi | 150 | 0.652 | [0.445, 0.757] | 0.357 | [-0.003, 0.610] |
| AZL 2015-16 | combined | 150 | 0.675 | [0.379, 0.824] | 0.570 | [0.243, 0.737] |

Script commit 8b87a8554b47171b68a28548876bd967a8fe9388; MLflow run 92ff36c081d042a3a2b2a1bddc13d8ee.
