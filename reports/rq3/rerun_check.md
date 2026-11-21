# Step 3.5 / 3.1 Check for RQ3: identical pool rebuild

The pool was built twice from `configs/experiments/rq3.yaml` at commit `c3279c0` (the second build into a scratch folder) and compared with `shortcutcheck experiment compare`. Columns include each model's validation F1, accuracy and a SHA-256 hash of its validation probabilities (MLflow run IDs ignored). No test data was opened.

`reports/rq3/pool_models.csv` vs `/tmp/pool-b/pool_models.csv`:

- pool_models.csv: 300/300 rows identical over 16 columns (largest difference 0) -> IDENTICAL

Verdict: **IDENTICAL**

