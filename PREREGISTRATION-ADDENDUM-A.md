# Pre-registration addendum A: explaining the results (Phase A)

Nisula Dasanayake · 21 November 2026

This addendum is committed **before** any of the three analyses below is computed. It does not
change H1–H3 or their verdicts, which stand as reported (H1 partially supported, H2 not
supported and reversed, H3 "No signal"). It adds analyses that explain those results. The code
lives in `src/shortcutcheck/addenda/` (not in the pre-registered `analysis/` package) and
`shortcutcheck addendum a1|a2|a3` refuses to run unless this file is committed and unchanged.

**What was known when this was written.** All of the H1–H3 results in `reports/rq1`, `rq2`
and `rq3`, and the exploratory RQ2 breakdown (`reports/rq2/exploratory.md`: debiased DR models
flip invalid-year inputs at 0.20–0.29). A1 and A2 were designed after seeing those results.
They are therefore **explanatory analyses with pre-stated decision rules**, not independent
confirmations. A2 in particular uses data whose results were already seen. The AndroCT
2015–16 holdout (`rq3-AndroCT-holdout-2015-16`, frozen in ddfe547) is **not used here**; it is
reserved for a later addendum.

## A1. Why did debiased training increase metadata reliance? (H2 mechanism)

**Data:** the training partitions of `rq12-DR-AG`, `rq12-DR-AG_deb`, `rq12-VS-AG` and
`rq12-VS-AG_deb` only (training labels; no test data).

**Measure:** the predictiveness of a feature group = mean ROC-AUC of 5-fold stratified
cross-validation (seed 11) of a LightGBM model (`build_model("lgbm", 11)`, the RQ1 default)
trained on that group alone. Three groups:

- **M1, raw metadata:** `year`, `apk_size` (the raw values the models use).
- **M2, Dada's binned proxies:** `Year`, `APK size` (the binned columns Dada balanced; D4
  dropped them from every model).
- **M3, behaviour:** every model feature that is not metadata (`year`, `apk_size`,
  `minSdkVersion`, `targetSdkVersion`).

**Size control:** a biased training set is much larger than its debiased counterpart, and
cross-validated AUC depends on size. So each biased set is measured on stratified subsamples
of the debiased set's size (seeds 11, 22, 33, 44, 55; mean reported, as in D17). Debiased sets
are measured whole.

**Two competing explanations, judged separately per family (DR, VS); both, one or neither may hold:**

- **(a) Proxy debiasing leaves the raw shortcut.** Holds if M2 on the debiased set ≤ 0.60
  (Dada's binned proxies carry little signal) **and** M1 on the debiased set ≥ M1 on the
  size-matched biased set − 0.02 (the raw metadata stays as predictive, or more).
- **(b) Weaker behaviour signal.** Holds if M3 on the debiased set ≤ M3 on the size-matched
  biased set − 0.02 (behaviour features carry less signal after debiasing, leaving metadata
  relatively more useful).

0.02 AUC is the margin for "no real difference". It is about twice the typical seed-to-seed
standard deviation of cross-validated AUC at these sizes. Also reported, without a verdict:
M1 − M3 (metadata advantage relative to behaviour) per set, and the same measures with
logistic regression.

## A2. Does the violation rate measure real metadata reliance? (convergent validity)

**Data:** the RQ1 results already computed (`reports/rq1/rows.csv`, `detection.csv`; DR-AG and
VS-AG test partitions). No new data is opened.

**Unit:** one (split, algorithm, seed), 50 in total. **Violation:** the realism-filtered MR4
rate of the full-feature model (stratum "all"). **Metadata gain:** test F1 of the full-feature
model minus test F1 of the behaviour-only model with the same split, algorithm and seed.

**Rule:** the test measures real reliance if Spearman ρ(violation, metadata gain) ≥ 0.5 **and**
the 95% cluster-bootstrap CI (2,000 resamples of the 10 split × algorithm configurations,
seed 11) has a lower bound > 0. With only 10 clusters the CI will be wide; that is reported
openly. Also reported, without a verdict: MR1 instead of MR4, and ROC-AUC gain instead of F1
gain.

## A3. Plain-Gower realism sensitivity (promised in PREREGISTRATION.md section 5)

The original pre-registration listed this as a secondary analysis, and it was not computed in
step 3.8. For all 300 RQ3 pool models, the pre-deployment MR4 rate is recomputed on the
2012–14 validation partition with the **plain Gower** realism check (equal weight per column;
`RealismCheck(metadata_columns=None)`; same seed, percentile and reference sample). No test
data is opened. Reported **without a verdict**: ρ with the F1 drop on the un-injected subset
and the full pool, with cluster-bootstrap CIs, next to the weighted-Gower result (0.084 and
−0.202). The H3 verdict is not changed by this.
