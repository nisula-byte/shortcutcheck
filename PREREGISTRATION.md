# Pre-registration: ShortcutCheck RQ1–RQ3 (roadmap step 3.2)

Nisula Dasanayake · 21 November 2026

This file fixes the hypotheses, the data each one is tested on, the exact tests and the
decision rules for RQ1–RQ3 **before any test partition is opened**. The analysis code committed
together with this file (`src/shortcutcheck/analysis/`) implements these rules and is the
pre-registered version.

Three checks make this binding:

1. `shortcutcheck prereg-check` (also run by `make rq1`, `make rq2` and `make rq3`) fails unless
   this file was committed before every entry in `logs/test_access.jsonl`. That includes the
   first access to the 2017–19 test partition, which is the roadmap's Check for step 3.2.
2. Every hypothesis report (`reports/rq1/h1.md`, `reports/rq2/h2.md`, `reports/rq3/h3.md`) states
   whether the analysis code still equals the version in this commit. If it has changed,
   rule 3 applies: the result is reported under both versions.
3. The experiment settings live in `configs/experiments/rq1.yaml`, `rq2.yaml` and `rq3.yaml`, in
   this commit. A changed setting is a deviation, logged with its reason and reported next to
   the pre-registered result.

## 1. What was known when this was written

Stated openly, because the H1 thresholds are set after the pilot (the roadmap's order):

- **Pilot RQ1** (step 2.7, `reports/pilot_rq1_rq12-DR-AG.md`): 50 models on the DR-AG
  *validation* partition. Mean violation rates, full features (behaviour-only models were
  exactly 0 on MR1, MR3 and MR4):

  | Algorithm | MR1 | MR3 | MR4 | MR5 | MR6 | F1 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Logistic regression | 0.002 | 0.005 | 0.061 | 0.103 | 0.039 | 0.962 |
  | Linear SVM | 0.002 | 0.005 | 0.059 | 0.094 | 0.023 | 0.961 |
  | Random forest | 0.044 | 0.001 | 0.094 | 0.001 | 0.001 | 0.988 |
  | LightGBM | 0.284 | 0.002 | 0.309 | 0.002 | 0.003 | 0.984 |
  | MLP | 0.003 | 0.001 | 0.027 | 0.101 | 0.030 | 0.979 |

- **Step 1.8 baselines:** validation-partition detection scores of random forest and logistic
  regression on all five splits, including VS-AG. No MR has been run on VS-AG, DR-AG_deb,
  VS-AG_deb or AndroCT models.
- **Steps 2.4–2.5:** realism exclusion rates and known-answer results on all splits. These use
  oracle models and validation inputs, not the models tested here.
- **2017–19 data:** the step 1.4 audit counted apps per class and year in that window, to check
  that RQ3 has enough data. No model has been trained on or scored on these apps.
- **Test partitions:** none opened. `logs/test_access.jsonl` does not exist at this commit.

## 2. Common settings (all hypotheses)

- **Data:** the frozen splits of step 1.7 (`data/splits/`, D5, D1). Models train on the training
  partition only. RQ1 and RQ2 are evaluated on the **test** partitions, so the pilot's
  validation inputs are not reused.
- **Models:** logistic regression, linear SVM (sigmoid-calibrated, 3-fold), random forest,
  LightGBM and MLP, with the fixed untuned settings of `training/baselines.build_model`. Seeds
  are 11, 22, 33, 44 and 55.
- **MRs:** MR1 (year), MR3 (APK size) and MR4 (joint metadata swap) are the core MRs. MR2 does
  not apply to any Dada split (D10). MR5 and MR6 (permissions, D12) are secondary and never
  decide a hypothesis. MRs and the realism check are fitted on the training partition of the
  split whose inputs are scored.
- **Counting:** an input counts only if it was perturbed, is valid and passes the realism check
  (D13). Each per-model rate has a 95% CI: percentile bootstrap with 2,000 resamples, or Wilson
  when 0 or all inputs violate (D14). Prediction is single-threaded.
- **Inputs:** a model is never scored on an app it was trained on. When a partition has more
  than 5,000 apps (VS-AG: 20,903), MRs run on one uniform random sample of 5,000 (seed 11). The
  sample is drawn without labels and shared by every model. Detection quality uses all inputs.
- **Test access:** each experiment opens each test partition once through `load_test()`. Labels
  are used only for the detection-quality tables (RQ1, RQ2) and the F1 drop (RQ3).

## 3. H1 (RQ1, step 3.3; `make rq1`, `analysis/h1.py`)

**Proposal wording:** "On biased training data, perturbing time and size metadata alone flips
at least 5% of predictions. A model trained without those features flips under 1%." The
proposal allowed the thresholds to be adjusted after the pilot, with reasons.

**Operational form.** Biased splits DR-AG and VS-AG, full-feature models, core MRs, test
partitions, realism filter on.

- An algorithm **shows the shortcut** on a split if, for at least one core MR, (a) its mean
  violation rate over the 5 seeds is **≥ 0.05**, and (b) the 95% CI lower bound is **> 0.01 for
  every seed**. A seed with no counted input fails (b).
- **H1a** holds on a split if **at least 3 of the 5 algorithms** show the shortcut.
- **Control:** behaviour-only models must be **exactly 0** on every core MR, for every
  algorithm and seed: no violation and no change in p. If not, the runner is broken and H1 is
  not judged.
- **Verdict:** SUPPORTED if H1a holds on both splits; PARTIALLY SUPPORTED if on one; NOT
  SUPPORTED if on neither.

**Changes from the proposal, and why:**

1. "Time and size metadata" means MR1, MR3 and MR4, because MR2 cannot run on Dada (D10).
2. The 5% target is kept unchanged. Rule (b) and the 3-of-5 rule are added. Rule (b) rules out
   an effect that rests on one lucky seed or a handful of inputs. The 3-of-5 rule means the
   effect is not specific to one model family. Both are the Gate 2 rules from
   `docs/pilot_plan.md`, plus the 5% size requirement.
3. "Under 1%" becomes "exactly 0", and it is a **validity control, not evidence for H1**. A
   model without metadata features cannot react to metadata-only changes, so 0 holds by
   construction. Claiming it as a finding would be circular.

**How the pilot scores under these rules:** 4 of 5 algorithms show the shortcut on DR-AG
validation data: logistic regression (MR4), linear SVM (MR4), random forest (MR4) and LightGBM
(MR1, MR4). The MLP does not (MR4 mean 0.027). **Caveat:** the DR-AG models in RQ1 are the
pilot models. Training is deterministic and uses the same data and seeds, and only the inputs
change from validation to test. The DR-AG result is therefore close to a replication of the
pilot. **VS-AG is the new test**, with 13 times more training data, a different malware source
(VirusShare) and no MR result seen yet.

**If H1 fails:** the RQ1 result is reported as it is. The thresholds and MRs are not changed
to make it pass (rule 3). The project then follows the Gate 3 route in the roadmap.

## 4. H2 (RQ2, step 3.4; `make rq2`, `analysis/h2.py`)

**Proposal wording:** "Debiased training lowers the violation rate significantly (Wilcoxon over
seeds, p < 0.05, Holm-corrected)."

**Design** (`configs/experiments/rq2.yaml`). Full-feature models in two families: DR-AG vs
DR-AG_deb, and VS-AG vs VS-AG_deb. Each family has three arms, matched by algorithm and seed:

- **published:** the biased training set as published;
- **size-matched:** the biased training set cut, stratified and with a different sample per
  seed, to the debiased set's size (DR: 7,424 → 1,157 apps; VS: 97,545 → 3,377 apps);
- **debiased:** the debiased training set.

All arms of a family are scored on **the same inputs**, with every app that any arm trained on
removed. The splits share apps; for example, 76 of the 249 DR-AG_deb test apps and 217 of the
724 VS-AG_deb test apps are in the matching biased training set. The input sets are:

- `h2-deb` (**primary**): the debiased split's test partition;
- `h2-biased` (secondary): the biased split's test partition.

**Test.** For each family, input set and core MR, d = rate(biased) − rate(debiased) over the
**25 pairs** (5 algorithms × 5 seeds). The test is a **one-sided Wilcoxon signed-rank test**
for d > 0 (SciPy: `zero_method="wilcox"`, `method="auto"`; all-zero differences give p = 1).
The two-sided p-value is reported next to it. **Holm** correction runs across the core-MR tests
of both families, separately for each (comparison, input set). **Cliff's delta** between the 25
biased and 25 debiased rates is reported next to every p-value.

**Decision rules (α = 0.05):**

- A family **supports H2** if MR4 (published vs debiased, `h2-deb`) has Holm p < 0.05.
- **Verdict:** SUPPORTED if both families support it; PARTIALLY SUPPORTED if one; NOT
  SUPPORTED if neither.
- **Robust to training size:** per family, the same MR4 test with the size-matched arm in
  place of the published arm is also significant. If not, the reduction cannot be separated
  from the smaller training set, and the report says so.
- MR1, MR3 and the `h2-biased` inputs are reported, not judged. The cross-family runs (each
  model scored on the other family's test partition) are descriptive.

**Changes from the proposal, and why:**

1. **Pairs over algorithm × seed, not over seeds.** With 5 pairs, the smallest possible
   Wilcoxon p-value is 0.031 one-sided and 0.0625 two-sided, so a per-algorithm test could
   barely or never reach p < 0.05 after Holm. Pairing over 25 algorithm-seed pairs follows the
   roadmap ("matched by algorithm and seed").
2. **One-sided,** because H2 predicts a direction. The two-sided p-value is also reported.
3. **MR4 is the deciding MR** (the joint metadata swap). It is the only core MR that every
   algorithm reacted to in the pilot. MR3 was at most 0.005 everywhere, so it would only
   dilute a combined rule.
4. **Identical inputs, the exclusion of trained-on apps and the size-matched arm** are
   additions. Without them, a difference could come from different inputs, from memorised
   apps, or simply from 6 to 29 times less training data.

## 5. H3 (RQ3, steps 3.5–3.8; `make rq3`, `analysis/h3.py`)

**Wording (Revision v2, approved 20 Nov 2026):** H3a: on the un-injected subset, Spearman ρ
between the pre-deployment MR violation rate and the F1 drop is ≥ 0.5, with a cluster-bootstrap
95% CI above 0. H3b: ρ(MR) > ρ(SHAP metadata share), paired cluster bootstrap on the
difference, 95% CI above 0. H3c (secondary): ρ(MR, pre-deployment) ≥ ρ of each monitoring
baseline, with Holm across the H3b and H3c comparisons.

**Data.** `rq3-AndroCT-2012-14-vs-2017-19` (D1): training 10,260 apps and validation 2,565 apps
from 2012–14; test 9,818 apps from 2017–19.

**Model pool** (`configs/experiments/rq3.yaml`; full-feature models only, because
behaviour-only models are 0 on MR1–MR4 by construction):

- 5 algorithms × 5 seeds. Skew level 0 uses 6 hyperparameter settings; levels 0.25, 0.5 and
  0.75 use settings 0–1. That gives 60 configurations and 300 models; the un-injected subset is
  30 configurations and 150 models (D8).
- **Hyperparameter settings** (setting 0 = the RQ1 defaults):

  | Algorithm | 0 | 1 | 2 | 3 | 4 | 5 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Logistic regression | default (C = 1) | C = 0.01 | C = 0.001 | C = 0.1 | C = 10 | C = 100 |
  | Linear SVM | default (C = 1) | C = 0.01 | C = 0.001 | C = 0.1 | C = 10 | C = 100 |
  | Random forest | default | max_depth 8 | max_depth 4 | min_samples_leaf 20 | max_features 0.5 | max_features log2 |
  | LightGBM | default | num_leaves 7 | num_leaves 127 | learning_rate 0.03 | min_child_samples 100 | colsample_bytree 0.5 |
  | MLP | default (100) | hidden (16) | hidden (256) | hidden (64, 64) | alpha 0.01 | alpha 1.0 |

  The settings span model capacity and regularisation, so that shortcut reliance varies between
  configurations. They are not tuned.
- **Skew injection** on `apk_size`. *Contrary* apps are malware below the training median
  (2,082,823.5 bytes) or goodware at or above it. Level q removes a random fraction q of each
  class's contrary apps (seed = model seed). Every level is then cut, stratified and seeded, to
  the per-class sizes left at q = 0.75: **3,286 goodware and 3,013 malware (6,299 apps)**. Only
  the skew differs between levels, not the amount of data. Size is used, not year, because year
  is also the time axis of RQ3.

**Signals.** Each is oriented so that a larger value means more decay is expected.

- **MR4 pre-deployment (method under test):** the realism-filtered MR4 violation rate on the
  2012–14 validation partition, with no labels.
- **SHAP metadata share (H3b competitor):** the mean over inputs of Σ|SHAP| on metadata columns
  divided by Σ|SHAP| on all columns. It uses 500 validation inputs and 100 training apps as
  background, both uniform samples with seed 11. Explainers: TreeExplainer (interventional, on
  the predicted probability) for the random forest and LightGBM; LinearExplainer (log-odds) for
  logistic regression; PermutationExplainer (predicted probability) for the linear SVM and the
  MLP.
- **Monitoring baselines (H3c)**, on 2017–19 test inputs with labels unused:
  - mean confidence drop: the mean of max(p, 1−p) on validation minus the same on test;
  - PSI of the predicted malware probability: 10 bins with edges at the validation deciles,
    duplicate edges merged, and bin shares floored at 0.0001;
  - ATC predicted accuracy drop: a threshold t on max(p, 1−p) is chosen so that the share of
    validation inputs above t equals validation accuracy (validation labels only). The signal
    is validation accuracy minus the share of test inputs above t.
- **Ground truth:** F1 drop = F1 on 2012–14 validation − F1 on 2017–19 test (threshold 0.5).
  Each model has one `load_test()` call (step 3.7).

**Analysis.**

- The unit is the model. Spearman ρ uses average ranks for ties. The cluster bootstrap draws
  **configurations** (algorithm × skew × setting) with replacement, 2,000 resamples, seed 11, and
  all 5 seeds of a drawn configuration come with it. CIs are percentile 95% intervals. A
  resample where a signal is constant gives no ρ, and the number of valid resamples is reported.
- **H3a:** on the un-injected subset, ρ(MR4, F1 drop) ≥ 0.5 **and** CI lower bound > 0.
- **H3b and H3c:** for each competitor k (SHAP share; confidence drop, PSI, ATC), Δk =
  ρ(MR4) − ρ(k) on the same resamples. The one-sided bootstrap p-value is
  (1 + #{Δk* ≤ 0}) / (1 + B). Holm runs across the 4 comparisons. A comparison holds if its
  Holm p < 0.05. H3b is the SHAP comparison; H3c is the other three.
- **Gate 3 outcome** (roadmap decision diagram, computed in code):
  - *Strong signal:* H3a and H3b hold.
  - *Signal, no edge:* H3a holds and H3b does not.
  - *Injected only:* H3a fails on the un-injected subset, but the same rule holds on the full
    pool.
  - *No signal:* none of the above.
- **Missing values:** a model with a missing signal or F1 drop is dropped from every comparison
  (so all signals use the same models), and the count is reported.
- **Secondary (reported, no verdict):** MR1, MR3, the core-MR mean, unfiltered MR4 (the realism
  ablation), MR4 measured on 2017–19 inputs, the plain-Gower realism variant (D13 sensitivity),
  the full pool, and the F1 drop per test year 2017, 2018 and 2019 (D9).

**Changes from Revision v2, and why:**

1. **MR4 is the single primary MR signal.** Revision v2 said "MR violation rate" without naming
   an MR, and several candidates would multiply the tests. MR4 changes all metadata together and
   is the contribution claim. MR1 can only move years within 2012–14 on this split (three
   training years), so it is secondary. The pilot used no AndroCT data, so this choice is not
   informed by RQ3 data.
2. **Linear SVM uses PermutationExplainer, not LinearExplainer.** The calibrated SVM averages
   three sigmoid-calibrated linear models, so its output is not one linear function of the
   inputs.
3. **H3c is tested as superiority (Δ > 0), not "≥".** "≥" cannot be tested without a
   non-inferiority margin, which was never set. Superiority is the stricter reading.
4. **The skew injection and the hyperparameter settings are now exact.** Revision v2 named only
   the number of levels and settings.

**Implementation order.** The pool, signal and ground-truth code (steps 3.5–3.7) is written
after this commit, but before any 2017–19 test access. It must follow the definitions above and
in `rq3.yaml`. Any difference found while implementing is logged as a decision, with its reason,
before the 2017–19 partition is opened, and is reported.

## 6. Deviations from the proposal (for the supervisor)

| Item | Proposal | Pre-registered | Reason |
| --- | --- | --- | --- |
| H1 MRs | time and size MRs | MR1, MR3, MR4 | MR2 not applicable on Dada (D10) |
| H1 rule | ≥ 5% flips | ≥ 5% mean, CI > 1% on every seed, 3 of 5 algorithms, both biased splits | one-seed and one-model effects excluded |
| H1 control | < 1% | exactly 0, validity check only | holds by construction |
| H2 test | Wilcoxon over seeds | one-sided Wilcoxon over 25 algorithm-seed pairs, MR4 deciding | n = 5 cannot reach p < 0.05 after Holm |
| H2 inputs | not specified | identical inputs per family, trained-on apps removed, size-matched arm | separates debiasing from inputs, memorisation and data size |
| RQ3 window | 2014 → 2017 | 2012–14 → 2017–19 | 2014 vs 2017 too small (D1) |
| RQ3 pool | 200 models | 300 models, 30 un-injected configurations | power for H3b (D8) |
| H3 MR signal | MR violation rate | MR4 rate | one primary test |
| H3c | ≥ baselines | superiority, Holm with H3b | testable without a margin |

## 7. Threats named in advance

The H1 thresholds were set with the pilot in view (section 1), and the DR-AG RQ1 models are the
pilot models. `year` is both the time axis of RQ3 and the MR1 target. Only one decay setting is
used (AndroCT). The realism weights are a design choice (D13). Permission levels are current
ones (D12). Random-forest numbers differ in the last digits across CPU architectures (D15).
Drift detectors that need extra models (Transcendent, CADE, DREAM) are not compared.

## 8. References

[1] F. Pendlebury et al., "TESSERACT: Eliminating experimental bias in malware classification
across space and time," in *Proc. USENIX Security*, 2019.

[2] T. Miranda et al., "Debiasing Android malware datasets: How can I trust your results if
your dataset is biased?," *IEEE Trans. Inf. Forensics Security*, vol. 17, 2022.

[3] S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur and H. Sedghi, "Leveraging unlabeled
data to predict out-of-distribution performance," in *Proc. ICLR*, 2022.

[4] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*,
vol. 6, no. 2, pp. 65–70, 1979.

[5] N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," *Psychol.
Bull.*, vol. 114, no. 3, pp. 494–509, 1993.

[6] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in
*Proc. NeurIPS*, 2017.
