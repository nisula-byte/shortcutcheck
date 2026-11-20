# Phase 1 baselines: `rq12-VS-AG_deb` (roadmap step 1.8)

Validation partition only; test set untouched. Git commit `354af2240f5e31cd4a47a7be0b79a901e016c31f`. Per-run MLflow IDs are in the CSV next to this file.

| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |
| --- | --- | --- | --- | --- | --- |
| lr | behaviour | 0.760 ± 0.000 | 0.850 ± 0.000 | 0.882 ± 0.000 | 0.543 ± 0.000 |
| lr | full | 0.765 ± 0.000 | 0.851 ± 0.000 | 0.879 ± 0.000 | 0.561 ± 0.000 |
| rf | behaviour | 0.825 ± 0.006 | 0.905 ± 0.001 | 0.920 ± 0.001 | 0.463 ± 0.017 |
| rf | full | 0.863 ± 0.004 | 0.937 ± 0.001 | 0.951 ± 0.001 | 0.247 ± 0.018 |

## Shuffled-label check (permutation test)

20 random forests trained on shuffled training labels, scored on validation. The check is judged on mean ROC-AUC, which must sit within the margin of 0.5; margin = max(configured floor, 3 × standard error of the mean over permutations). PR-AUC is reported against the original criterion but not judged (decision D7: it is not centred on the prevalence when one class is more spread out in feature space).

| Measure | Mean | Chance | Margin | Range over permutations | Within margin | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC | 0.488 | 0.500 | 0.050 | 0.407–0.557 | yes | yes |
| PR-AUC | 0.386 | 0.390 | 0.050 | 0.332–0.421 | yes | no |

**Check: PASS.** Shuffled-label models score at chance on average.
