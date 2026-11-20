# Pilot RQ1 plan (roadmap step 2.7): stated before running

This file is committed before the pilot is implemented or run. Gate 2 is judged against the
criteria below exactly as written. If a criterion turns out to be badly chosen, the result
under this version is still reported, and any change is logged as a new decision with its
reason.

## Setting

- **Data:** frozen split `rq12-DR-AG` (Drebin malware + AndroZoo goodware, the biased
  "Drebin mixed set"). Models train on the training partition (7,424 apps). MRs are run on
  the whole validation partition (1,592 apps). The test partition is not touched.
- **Models:** 5 algorithms (logistic regression, linear SVM, random forest, LightGBM, MLP)
  × 2 feature sets (full; behaviour-only, i.e. metadata removed) × 5 seeds (11, 22, 33, 44,
  55) = 50 models. Settings are fixed in code before the run and never tuned: library
  defaults plus balanced class weights (as in step 1.8), feature scaling for the linear
  models and the MLP, and early stopping for the MLP. The linear SVM gets probabilities from
  sigmoid calibration (3-fold, training data only).
- **MRs:** every MR that applies to DR-AG: MR1, MR3 and MR4 (core; MR2 is not applicable,
  D10), MR5 and MR6 (secondary). Fitted on the training partition. Each model and MR uses
  the model's seed.
- **Counting:** the realism filter is on (D13). Only perturbed, valid and realistic inputs
  count. The unfiltered rate is reported next to it.
- **Statistics:** per model, the violation rate with a 95% bootstrap CI (Wilson when there
  are 0 or only violations) and the mean |change in p| with a bootstrap CI (D14). MR1 and
  MR4 are also reported separately for valid-year and invalid-year inputs (D11).
- **Detection quality** (F1, PR-AUC, ROC-AUC, FPR at 95% TPR) is reported per model on the
  validation partition, so violation rates can be read next to accuracy.
- **Traceability:** one MLflow run per model, tagged with the git commit. One command
  regenerates everything: `make pilot`.

## Gate 2 criteria (from the roadmap, made measurable here)

1. **Known-answer tests pass.** Already met in step 2.5 (`reports/known_answers.md`, 83 of 83).
2. **Behaviour-only models show near-zero violations on MR1–MR4.** They cannot see metadata,
   so the only acceptable value is **exactly 0** on MR1, MR3 and MR4, for every algorithm and
   seed. Anything else means the runner is wrong.
3. **Biased full-feature models show clearly non-zero violations.** An algorithm counts as
   *clearly non-zero* on an MR if, **for all 5 seeds**, the lower end of the 95% CI of its
   violation rate is above **0.01** (1%). The criterion is met if **at least 3 of the 5
   algorithms** are clearly non-zero on **at least one core MR** (MR1, MR3 or MR4). MR5 and
   MR6 are reported but do not count towards Gate 2.

Reasons for the numbers. 1% separates a real effect from a rounding error, and it is a low
bar: Gate 2 asks whether the effect exists, not how large it is (that is RQ1 in step 3.3).
Requiring every seed rules out one lucky seed. Requiring 3 of 5 algorithms means the effect
is not specific to one model family.

## What happens if a criterion fails

- **Criterion 2 fails:** the runner is broken. Fix it, re-run the known-answer tests and the
  pilot, and report both runs.
- **Criterion 3 fails:** H1 has failed in the pilot. As the roadmap says, discuss reframing
  with the supervisor before scaling up. Do not change the threshold, the MRs or the models
  until it passes without reporting this run.

## Not decided by this pilot

H1's thresholds for the full RQ1 run are pre-registered separately in step 3.2. In the
roadmap's order the pilot comes first, so those thresholds are set with the pilot in view.
`PREREGISTRATION.md` will say so openly and quote the pilot numbers. The full RQ1 run covers
all four RQ1/RQ2 splits, and only DR-AG is in the pilot.
