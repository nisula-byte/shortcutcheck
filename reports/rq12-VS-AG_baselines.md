# Phase 1 baselines: `rq12-VS-AG` (roadmap step 1.8)

Validation partition only; test set untouched. Git commit `354af2240f5e31cd4a47a7be0b79a901e016c31f`. Per-run MLflow IDs are in the CSV next to this file.

| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |
| --- | --- | --- | --- | --- | --- |
| lr | behaviour | 0.913 ± 0.000 | 0.971 ± 0.000 | 0.969 ± 0.000 | 0.143 ± 0.000 |
| lr | full | 0.913 ± 0.000 | 0.971 ± 0.000 | 0.970 ± 0.000 | 0.152 ± 0.000 |
| rf | behaviour | 0.945 ± 0.000 | 0.989 ± 0.000 | 0.988 ± 0.000 | 0.058 ± 0.001 |
| rf | full | 0.950 ± 0.000 | 0.990 ± 0.000 | 0.989 ± 0.000 | 0.048 ± 0.001 |

## Shuffled-label check (permutation test)

20 random forests trained on shuffled training labels, scored on validation. The check is judged on mean ROC-AUC, which must sit within the margin of 0.5; margin = max(configured floor, 3 × standard error of the mean over permutations). PR-AUC is reported against the original criterion but not judged (decision D7: it is not centred on the prevalence when one class is more spread out in feature space).

| Measure | Mean | Chance | Margin | Range over permutations | Within margin | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC | 0.501 | 0.500 | 0.050 | 0.492–0.511 | yes | yes |
| PR-AUC | 0.536 | 0.483 | 0.050 | 0.530–0.547 | no | no |

**Check: PASS.** Shuffled-label models score at chance on average.
