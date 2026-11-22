# Pre-registration addendum D: confirmatory tests of the fix (RQ4) and of decay on the 2015–16 holdout (H5)

Nisula Dasanayake · 22 November 2026

This addendum is committed **before** any data it tests is opened. It does not change H1–H3
or addenda A–C. It turns the addendum C proof of concept (GO on validation data) into
confirmatory tests on data that addendum C did not use, and it uses the locked 2015–16
holdout for the first time.

## Why

Addendum C found, on validation partitions, that one counterfactual copy per training app
(the app's behaviour with the other class's metadata, label kept) cut MR4 violations by
70–98% while keeping F1. Validation results cannot be claimed: the fix must be confirmed on
unseen inputs (RQ4). The project's original aim was to anticipate temporal decay; H3 found no
signal before deployment, while MR4 measured at monitoring time correlated with decay
(ρ = 0.760, exploratory, 2017–19). The locked 2015–16 holdout allows two forward-looking tests
that were stated before any of its data is read: whether the fix reduces decay (H5a), and
whether MR4 at monitoring time predicts decay (H5b, H5c).

**Known before writing this:** every H1–H3 and addendum A–C result. The RQ1/RQ2 test
partitions were opened before (RQ1, RQ2); the baseline arm below uses the same training data,
algorithms and seeds as the RQ1/RQ2 full-feature models, so its test numbers are largely known.
No result of the augmented or duplicate-control models on any test partition exists. The
2017–19 test partition was opened before (RQ3); its labels are known for the RQ3 pool, not for
the models trained here. The 2015–16 holdout (`rq3-AndroCT-holdout-2015-16`, 7,129 apps:
4,551 goodware, 2,578 malware; frozen ddfe547) has **never been opened** (0 access-log
entries). The fix is exactly addendum C's (D21); nothing is tuned.

## Common design

- **Fix (augmented arm):** one MR4 copy per training app with the other class's metadata
  relative to its true label, label kept, kept only if valid and realistic (block-weighted
  Gower, D13, fitted on the split's training partition, seed 11); drawn with the model's seed
  (`addenda/fix_poc.augment`, unchanged).
- **Duplicate control:** every training app twice, unchanged. It separates the effect of the
  metadata change from the effect of a larger, repeated training set.
- **Models:** full features; LR, SVM, RF, LightGBM, MLP (`build_model`, untuned); seeds 11, 22,
  33, 44, 55. Pairs: 25 per split (algorithm × seed). Prediction single-threaded.
- **MRs and realism:** fitted on the split's training partition; realism filter on; MR inputs
  are the evaluation partition, or a uniform 5,000-app sample of it (seed 11) when larger.
- **Contrary-app accuracy:** accuracy on the evaluation apps that a metadata-only LightGBM
  (`year`, `apk_size`; trained on the training partition, seed 11) misclassifies.
- **Data access:** every test partition and the holdout are opened once, through `load_test()`,
  with the purpose logged. MRs and label-free signals never see labels; labels are used only for
  F1, accuracy and contrary-app accuracy.

## RQ4 / H4: does the fix hold on unseen test inputs?

- **Splits (train on the training partition, evaluate on the test partition):** `rq12-DR-AG`,
  `rq12-DR-AG_deb`, `rq12-VS-AG`, `rq12-VS-AG_deb`.
- **Arms:** baseline, augmented, and duplicate control (all splits except `rq12-VS-AG`, whose
  97,545 training apps make a third arm too costly).
- **Rules per split** (d = baseline − augmented, 25 pairs):
  - **H4a (primary):** MR4 falls: one-sided Wilcoxon signed-rank (baseline > augmented),
    Holm over the 4 splits, adjusted p < 0.05, **and** mean augmented MR4 ≤ 50% of baseline.
  - **H4b (cost):** mean test F1 falls by at most 0.02.
  - **H4c (held-out MRs):** MR1 or MR3 falls: one-sided Wilcoxon per MR, Holm over every MR1
    and MR3 test in every split (up to 8); holds if either adjusted p < 0.05.
- A split **passes** if H4a, H4b and H4c hold. **H4 is SUPPORTED** if at least 3 of the 4 splits
  pass, **PARTIALLY SUPPORTED** if 1 or 2 pass, **NOT SUPPORTED** if none does.
- **H4d (judged, secondary): the fix helps where metadata misleads.** Contrary-app accuracy
  rises: one-sided Wilcoxon (augmented > baseline), Holm over the 4 splits. Reported as the
  number of splits where it holds.
- **Reported, not judged:** MR5, MR6 (permissions, D20); ROC-AUC; the duplicate control against
  baseline and against augmented for every measure; rejected copies; per-algorithm results.

## H5: forward-looking tests on the 2015–16 holdout

Training period 2012–14 (`rq3-AndroCT-2012-14-vs-2017-19`, training partition); reference F1
on its 2012–14 validation partition; the holdout 2015–16 is the deployment period.

**H5a (the fix reduces decay).** Arms baseline, augmented, duplicate control; 5 algorithms ×
5 seeds. Decay = F1(2012–14 validation) − F1(2015–16 holdout).
- **H5a holds** if decay(baseline) > decay(augmented): one-sided Wilcoxon over the 25 pairs,
  p < 0.05.
- Reported, not judged: holdout F1 per arm; augmented vs duplicate control; the same decay
  comparison on the 2017–19 test partition (opened once more in this addendum); MR4 on the
  holdout per arm.

**H5b (MR4 at monitoring time predicts decay).** The 300 models of the RQ3 pool (`data/models/
rq3`, rebuilt identically in step 3.5). Before the holdout is opened, every model's validation
probabilities are hashed and compared with `reports/rq3/pool_models.csv`; the run stops if any
differs. For each model, on the holdout inputs
(no labels): the MR4 violation rate on a uniform 5,000-app sample (seed 11) with the realism
filter, MRs and realism fitted on the 2012–14 training partition, seed = model seed
(`mr4_monitor`); and the monitoring baselines exactly as in step 3.6: PSI, mean confidence drop
and ATC predicted-accuracy drop (references from the 2012–14 validation partition). Then, with
the holdout labels: F1 on the holdout and decay = F1(validation) − F1(holdout).
- Analysis exactly as H3: Spearman ρ with decay; headline subset = the 150 un-injected models
  (30 configurations); cluster bootstrap over configurations, 2,000 resamples, seed 11;
  percentile 95% CIs.
- **H5b holds** if ρ(`mr4_monitor`, decay) ≥ 0.5 and the CI lower bound > 0 on the un-injected
  subset.
- **H5c (judged, secondary):** ρ(`mr4_monitor`) − ρ(b) > 0 for each baseline b in {PSI,
  confidence drop, ATC}: one-sided bootstrap p from the paired resamples, Holm over the 3
  comparisons, holds if adjusted p < 0.05.
- Reported, not judged: the full pool (300 models); `mr4_pre` (from step 3.6) against holdout
  decay, which re-tests H3 on a second deployment period.

## Order of work and access

1. This addendum is committed; the code is committed and passes its tests; a dry run with the
   same layout on validation partitions (no test access) completes.
2. One run. Expected test accesses, each logged once: the 4 RQ1/RQ2 test partitions, the
   2017–19 test partition, and the 2015–16 holdout (6 in total).
3. Reports are generated from the saved tables; every verdict is reported, whatever it is.

Families are separate: Holm applies within H4a, within H4c, within H4d and within H5c, not
across hypotheses.

## Deviations

Any change after this commit is listed here with its reason, and the report states it.

| Date | Change | Reason |
| --- | --- | --- |
