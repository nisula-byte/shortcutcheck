# Pre-registration addendum F: why the fix behaves like dropping metadata (mechanism)

Nisula Dasanayake · 22 November 2026

Committed **before** any quantity below is computed. Training and validation partitions only:
no test partition and no holdout is read. It does not change H1–H5 or addenda A–E.

## Why

Addenda D and E found that counterfactual metamorphic augmentation (the fix) reduces decay
mainly for linear models and the MLP, and that it matches, but does not beat, a model trained
without metadata (E5). This addendum tests an explanation for both findings.

**Known before writing this:** every H1–H5 and addendum A–E result. No quantity named below has
been computed for any model.

## Proposition (stated here, proved in the paper)

Let $P_y$ be the distribution of metadata $x_m$ among training apps of class $y$. If every app of
class $y$ receives exactly one copy whose metadata are drawn from $P_{1-y}$ (no copy dropped), the
metadata distribution of class $y$ in the augmented set is $\tfrac12 P_y + \tfrac12 P_{1-y}$, the
same for both classes. Hence in the augmented training set **metadata alone carry no information
about the label** (marginal independence), although metadata can still be informative *jointly
with* behaviour features, because each copy keeps its own behaviour. Consequence: a model that
combines features additively (LR, linear SVM) loses its incentive to weight metadata, which
approximates dropping them; a model that learns interactions (trees) can still use metadata
conditionally on behaviour. The realism filter drops some copies, so the independence is only
approximate in practice.

## Design (code `src/shortcutcheck/addenda/mechanism.py`; command `shortcutcheck addendum f`)

- **Splits:** `rq12-DR-AG`, `rq12-DR-AG_deb`, `rq3-AndroCT-2012-14-vs-2017-19`,
  `rq3-AZL-2012-14-vs-2017-19`; train on the training partition, measure on the validation
  partition.
- **Arms:** baseline, augmented (the fix exactly as in addendum C), behaviour-only; LR, SVM, RF,
  LightGBM, MLP (`build_model`, untuned); seeds 11, 22, 33, 44, 55.
- **Metadata reliance R** of a model: validation ROC-AUC minus the ROC-AUC after the four
  metadata columns are permuted jointly across validation apps; mean over 5 permutations
  (seed 11). Behaviour-only models have R = 0 by construction.
- **Agreement** with behaviour-only: share of validation apps on which a model's predicted label
  equals that of the behaviour-only model with the same algorithm and seed.
- **Metadata-only separability:** 5-fold stratified cross-validated ROC-AUC of a metadata-only
  LightGBM (seed 11) on the training set, before and after augmentation (augmentation seed 11).

## Predictions and rules

- **F1 (proposition holds in practice):** metadata-only cross-validated ROC-AUC on the augmented
  training set ≤ 0.60 in every split.
- **F2 (linear models stop relying on metadata):** for LR and for SVM, mean R(augmented) ≤ 0.5 ×
  mean R(baseline) (means over seeds), in at least 3 of the informative splits.
- **F3 (tree ensembles retain more):** the relative reduction 1 − R(augmented)/R(baseline),
  averaged over RF and LightGBM, is smaller than the same average over LR and SVM, in at least 3
  of the informative splits.
- **F4 (linear models converge to the metadata-free model):** for LR and SVM, agreement with the
  behaviour-only model is higher for augmented than for baseline models: one-sided Wilcoxon over
  the 10 algorithm–seed pairs per split, Holm over the splits; holds if adjusted p < 0.05 in at
  least 3 splits.

A split is **informative** for F2 and F3 only if the mean baseline R of the algorithms involved
is at least 0.005 (a smaller R cannot be halved meaningfully); non-informative splits are
reported and excluded from the count. If fewer than 3 splits are informative, F2 or F3 is
reported as NOT TESTABLE.

**The mechanism is supported** if F1, F2 and F4 hold. F3 is judged and reported separately
(it explains the algorithm pattern). MLP results are reported, not judged.

## Deviations

| Date | Change | Reason |
| --- | --- | --- |
