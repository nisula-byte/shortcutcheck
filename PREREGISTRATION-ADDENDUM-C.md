# Pre-registration addendum C: does metamorphic augmentation fix metadata reliance? (proof of concept)

Nisula Dasanayake · 22 November 2026

This addendum is committed **before** the code below runs. It is the cheap proof of concept
for a fix (recovery Phase B). It uses **training and validation data only**: no test
partition and not the locked 2015–16 holdout. It does not change H1–H3 or addenda A and B.
Its only purpose is a go/no-go decision: whether the fix is worth a later confirmatory test
(RQ4), which would get its own addendum and would run on data this addendum does not touch.

## Why

RQ1 found that models trained on biased data change their predictions when only metadata
changes (H1 partially supported). RQ2 and addendum B found that models trained on Dada's
debiased data change their predictions under metadata changes *more*, and (exploratory, D20)
are less stable in general. A model-level fix would be useful in both cases: if a detector
must not depend on metadata, teach it so directly. Metamorphic augmentation does this by
adding, for each training app, a copy that carries the metadata of a real training app from
the **other** class while keeping the original behaviour features and the original label.

**Known before writing this:** every H1–H3, addendum A and addendum B result. The validation
partitions used below have been read before (pilot on DR-AG; addendum A3 on the RQ3 split;
dry runs). No result of the augmented models exists. The fix's settings below are fixed now
and are not tuned.

## Design (code: `src/shortcutcheck/addenda/fix_poc.py`; command `shortcutcheck addendum c`)

- **Splits (training partition to train, validation partition to evaluate):**
  `rq12-DR-AG` (biased; H1 held), `rq12-DR-AG_deb` and `rq12-VS-AG_deb` (debiased; the
  unstable models of RQ2), `rq3-AndroCT-2012-14-vs-2017-19` (the RQ3 training period).
  `rq12-VS-AG` (97,545 training apps) is left out to keep the proof of concept cheap.
- **Models:** full features; LR, SVM, RF, LightGBM, MLP (`build_model`, untuned); seeds
  11, 22, 33. Two arms per split, algorithm and seed:
  - **baseline:** trained on the training partition;
  - **augmented:** trained on the training partition plus one augmented copy per training app.
- **The augmented copy (fixed, not tuned):** MR4 (full metadata swap) fitted on the training
  partition, applied to each training app with the *other class relative to its true label*
  as the donor class, so the copy wears the other class's metadata. The label is kept. A copy
  is kept only if it passes MR4's validity check and the realism check (block-weighted Gower,
  D13, fitted on the training partition, seed 11). Augmentation draws use the model's seed.
- **Evaluation (validation partition, all apps, no labels for MRs):** MR1, MR3, MR4, MR6
  (and MR5 where it applies) through `run_mr` with the realism check, seed = model seed,
  stratum "all"; detection F1; and **contrary-app accuracy**: accuracy on the validation apps
  that a metadata-only LightGBM (`year`, `apk_size`; trained on the training partition, seed
  11) misclassifies, i.e. the apps whose metadata points to the wrong class.
  Prediction is single-threaded.
- **Pairs:** 15 per split (5 algorithms × 3 seeds), baseline vs augmented with the same
  algorithm and seed. d = measure(baseline) − measure(augmented).

## Rules (decided now)

Per split:

- **(a) The trained-on MR falls:** mean MR4 rate of the augmented arm ≤ 50% of the baseline's.
  (Partly circular, since MR4 made the copies; a fix that fails this does not work at all.)
- **(b) Detection is kept:** mean validation F1 falls by at most 0.02.
- **(c) A metadata MR not used for augmentation falls:** one-sided Wilcoxon signed-rank
  (baseline > augmented) on MR1 and on MR3 over the 15 pairs; Holm over every such test in
  every split (up to 8). (c) holds in a split if at least one of its two Holm-adjusted p < 0.05.
  An MR that does not apply in a split is skipped there.

A split **passes** if (a), (b) and (c) all hold. Overall:

- **GO** if at least 3 of the 4 splits pass: write the RQ4 confirmatory addendum with this fix
  exactly as specified here.
- **PARTIAL** if 1 or 2 splits pass: report which; a confirmatory test may cover only the kind
  of split that passed (biased or debiased), and must say so.
- **NO-GO** if no split passes: the fix is reported as not working, and no RQ4 test is run.

**Reported, not judged:** MR6 and MR5 (permission MRs, not metadata; if they fall as much as
MR4, the fix improves stability in general rather than metadata reliance specifically, D20);
contrary-app accuracy change (two-sided Wilcoxon); ROC-AUC; the number of augmented copies
rejected by the validity and realism checks; per-algorithm results.

## Deviations

Any change after this commit is listed here with its reason, and the report states it.

| Date | Change | Reason |
| --- | --- | --- |
