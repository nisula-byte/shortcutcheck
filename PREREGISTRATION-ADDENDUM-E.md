# Pre-registration addendum E: replication of the decay results on an independent dataset (AZL)

Nisula Dasanayake · 22 November 2026

This addendum is committed **before** any data it tests is opened. It does not change H1–H5 or
addenda A–D. It answers three objections to addendum D: (1) the decay results rest on one dated
pool (AndroCT); (2) part of the H5b correlation is a difference between algorithms; (3) the fix
was not compared with the simplest alternative, a model that does not see metadata at all.

## Data (frozen before this addendum, never opened)

AZL from Dada v3.1 (`AZL-G` goodware, `AZL-M` malware, AndroZoo-sampled), deduplicated, valid
years only, and with **every app that occurs in the AndroCT pool removed** (99 apps), so the
replication shares no app with any earlier analysis (config candidate `AZL-2012-14-vs-2017-19`,
key `exclude_pools_of`).

| Split | Part | Apps (goodware / malware) | Use |
| --- | --- | --- | --- |
| `rq3-AZL-2012-14-vs-2017-19` | train (2012–14) | 5,466 (3,629 / 1,837) | training |
| | val (2012–14) | 1,367 (907 / 460) | reference F1, PSI/ATC references |
| | test (2017–19) | 2,534 (1,680 / 854) | secondary decay period |
| `rq3-AZL-holdout-2015-16` | test (2015–16) | 5,115 (3,773 / 1,342) | primary deployment period |

Both test parts are locked behind `load_test()`; neither has been read (access log: 316 lines,
none for AZL).

**Known before writing this:** every H1–H5 and addendum A–D result, including the exploratory
algorithm-adjusted H5b values on AndroCT (0.521 un-injected, 0.727 full pool) and the
per-algorithm H5a pattern (benefit concentrated in LR and SVM). No AZL model has been trained.
The fix is exactly addendum C's (D21); nothing is tuned.

## Design (code: `src/shortcutcheck/addenda/addendum_e.py`, runner
`src/shortcutcheck/evaluation/addendum_e_run.py`; command `shortcutcheck addendum e`)

Everything not stated here is as in addendum D (full features; untuned `build_model`; MRs and
realism fitted on the training part; uniform 5,000-app MR sample, seed 11; single-threaded
prediction; decay = F1(2012–14 validation) − F1(deployment period); labels used only after the
label-free signals are computed).

- **Pool (E2–E4):** 150 models = 5 algorithms × the 6 hyperparameter settings of
  `configs/experiments/rq3.yaml` (skew level 0) × seeds 11, 22, 33, 44, 55, trained on AZL
  2012–14. Configuration = algorithm × setting (30 configurations). Signals on the 2015–16
  inputs: `mr4_monitor` (MR4 rate with realism, seed = model seed), PSI (validation deciles,
  floor 1e-4), confidence drop, ATC drop, exactly as in step 3.6.
- **Arms (E1, E5):** baseline, augmented (the fix), duplicate control, and **behaviour-only**
  (the same algorithm trained on the 202 behaviour features only); 5 algorithms × seeds
  11–55 = 25 pairs per comparison.

## Hypotheses and rules

- **E1 (replicates H5a): the fix reduces decay.** decay(baseline) > decay(augmented) on 2015–16,
  one-sided Wilcoxon over 25 pairs, p < 0.05.
- **E2 (replicates H5b): MR4 at monitoring time predicts decay.** Spearman ρ(`mr4_monitor`,
  decay) ≥ 0.5 and 95% cluster-bootstrap CI lower bound > 0 (2,000 resamples over the 30
  configurations, seed 11).
- **E3 (new): the correlation holds within algorithms.** ρ_adj = Spearman correlation after
  removing algorithm means: rank both variables over all models, subtract each algorithm's mean
  rank, and take the Pearson correlation of the centred ranks. Holds if ρ_adj > 0 with the 95%
  CI lower bound > 0, the CI from the same configuration bootstrap with ranking and centring
  redone in every resample.
- **E4 (replicates H5c): MR4 beats the monitors.** ρ(`mr4_monitor`) − ρ(b) > 0 for each b in
  {PSI, confidence drop, ATC}: one-sided bootstrap p on the paired resamples, Holm over 3,
  adjusted p < 0.05.
- **E5 (new, two-sided): the fix vs a metadata-free model.** On 2015–16, compare augmented and
  behaviour-only models on holdout F1 and on decay: two-sided Wilcoxon over 25 pairs, Holm over
  the 2 tests. Reported as "fix better", "behaviour-only better" or "no difference" for each
  measure, by the sign of the mean difference where adjusted p < 0.05.

**Overall replication verdict (E1 and E2):** REPLICATES if both hold, PARTIALLY REPLICATES if
one holds, DOES NOT REPLICATE if neither holds. E3, E4 and E5 are judged and reported
separately.

**Reported, not judged:** every arm's F1 and decay on 2015–16 and on 2017–19; per-algorithm
results; ρ on the 2017–19 period; MR4 on the holdout per arm; the rejected augmentation copies.
Secondary (data seen before, **not confirmatory**): E3's adjusted ρ with CI recomputed on the
addendum D AndroCT table, and the E5 comparison on AndroCT 2015–16 with 25 newly trained
behaviour-only AndroCT models (this reopens the AndroCT 2015–16 holdout once, for those models
only).

## Order of work and access

1. This addendum, the frozen AZL splits and the code are committed; tests pass; a dry run with
   the same layout on the AZL validation partition (no test access) completes.
2. One run. Expected test accesses, each logged once: AZL 2015–16, AZL 2017–19, and AndroCT
   2015–16 (secondary E5).
3. Reports generated from the saved tables; every verdict reported, whatever it is.

## Deviations

Any change after this commit is listed here with its reason, and the report states it.

| Date | Change | Reason |
| --- | --- | --- |
