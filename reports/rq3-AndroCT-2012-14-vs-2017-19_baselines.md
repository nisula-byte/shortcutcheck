# Phase 1 baselines: `rq3-AndroCT-2012-14-vs-2017-19` (roadmap step 1.8)

Validation partition only; test set untouched. Git commit `354af2240f5e31cd4a47a7be0b79a901e016c31f`. Per-run MLflow IDs are in the CSV next to this file.

| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |
| --- | --- | --- | --- | --- | --- |
| lr | behaviour | 0.892 ± 0.000 | 0.953 ± 0.000 | 0.955 ± 0.000 | 0.201 ± 0.000 |
| lr | full | 0.891 ± 0.000 | 0.955 ± 0.000 | 0.955 ± 0.000 | 0.209 ± 0.000 |
| rf | behaviour | 0.921 ± 0.001 | 0.980 ± 0.000 | 0.977 ± 0.000 | 0.128 ± 0.005 |
| rf | full | 0.923 ± 0.002 | 0.980 ± 0.000 | 0.978 ± 0.000 | 0.120 ± 0.004 |

## Shuffled-label check (permutation test)

20 random forests trained on shuffled training labels, scored on validation. The check is judged on mean ROC-AUC, which must sit within the margin of 0.5; margin = max(configured floor, 3 × standard error of the mean over permutations). PR-AUC is reported against the original criterion but not judged (decision D7: it is not centred on the prevalence when one class is more spread out in feature space).

| Measure | Mean | Chance | Margin | Range over permutations | Within margin | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC | 0.498 | 0.500 | 0.050 | 0.463–0.513 | yes | yes |
| PR-AUC | 0.516 | 0.479 | 0.050 | 0.492–0.540 | yes | no |

**Check: PASS.** Shuffled-label models score at chance on average.
