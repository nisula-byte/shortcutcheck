# Phase 1 baselines: `rq12-DR-AG` (roadmap step 1.8)

Validation partition only; test set untouched. Git commit `354af2240f5e31cd4a47a7be0b79a901e016c31f`. Per-run MLflow IDs are in the CSV next to this file.

| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |
| --- | --- | --- | --- | --- | --- |
| lr | behaviour | 0.958 ± 0.000 | 0.990 ± 0.000 | 0.992 ± 0.000 | 0.036 ± 0.000 |
| lr | full | 0.962 ± 0.000 | 0.991 ± 0.000 | 0.992 ± 0.000 | 0.030 ± 0.000 |
| rf | behaviour | 0.985 ± 0.001 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.001 ± 0.000 |
| rf | full | 0.988 ± 0.000 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.000 ± 0.000 |

## Shuffled-label check (permutation test)

20 random forests trained on shuffled training labels, scored on validation. The check is judged on mean ROC-AUC, which must sit within the margin of 0.5; margin = max(configured floor, 3 × standard error of the mean over permutations). PR-AUC is reported against the original criterion but not judged (decision D7: it is not centred on the prevalence when one class is more spread out in feature space).

| Measure | Mean | Chance | Margin | Range over permutations | Within margin | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC | 0.499 | 0.500 | 0.050 | 0.446–0.551 | yes | yes |
| PR-AUC | 0.612 | 0.500 | 0.050 | 0.571–0.663 | no | no |

**Check: PASS.** Shuffled-label models score at chance on average.
