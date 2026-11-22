# Weekly log (rule 6)

One paragraph per week: what was done, what failed, what was decided, which Check passed.
Bring this and the gate checklist to every fortnightly supervisor meeting.

## Week 0 (w/c 16 Nov 2026)

Phase 1 scaffold created: pinned Python 3.11 environment, CI on Linux and Windows, and tested
commands for roadmap steps 1.3–1.8 (register, audit, validity, splits, baselines). Tests run on
synthetic Dada-like data only. The shuffled-label check was redesigned as a permutation test
(mean of 20 shuffles) after single shuffles proved to vary widely when one strong feature
dominates the data. Literature review Draft 1 written; idea revised to v2 (SHAP metadata share
and ATC added to RQ3), pending supervisor approval.

Later in Week 0: Revision v2 approved. Steps 1.3–1.8 run on the real Dada v3.1 data and merged
(PR #1): every Gate 1 criterion met (decisions D1–D9). Phase 2 started early: MR interface,
MR1–MR4 and runner skeleton (steps 2.1–2.2) with property-based tests. MR2 turns out not to be
applicable on Dada because SDK versions are missing (0) in over 99.9% of apps (D10). Apps with
invalid years (mostly 1980) are almost all goodware in DR-AG, so MR1 never creates them, and
violation rates will be reported separately for valid- and invalid-year inputs (D11, step 2.6).
Step 2.3 done the same day: MR5/MR6 permission pools taken only from the official protection
levels in AOSP's framework manifest (34 permissions: 11 normal for MR5, 14 dangerous for MR6, 9
excluded). MR5 is not applicable on VS-AG, where malware requests every normal permission more
often than goodware, and has only 2 permissions on AndroCT and VS-AG_deb (D12).
Step 2.4 (realism check): plain Gower distance turned out blind to metadata MRs (91-100% of
excluded perturbed apps were already atypical before perturbation), so metadata and behaviour
now weigh equally (D13), decided before any violation rate was computed. Calibration ~5% on
every split; MR4 excludes 12-27% and MR1 8-22% of perturbed apps, MR3/MR6 about 5%.
Steps 2.5-2.6: oracle models with exactly computed expected flip rates; 83 of 83 testable known
answers pass on the real splits. Found and fixed a reproducibility bug: parallel random-forest
prediction changed probabilities by up to 5.6e-17, so the runner now predicts single-threaded.
A year-only model flips 72% under MR1 on AndroCT, easing the Gate 2 worry about MR1's range.
Step 2.7 pilot RQ1 on DR-AG (plan and Gate 2 criteria committed first, docs/pilot_plan.md):
Gate 2 pilot criteria PASS. Behaviour-only models exactly 0 on MR1-MR4; all 5 algorithms
clearly non-zero on a core MR (MR4 for all; MR1 for RF and LightGBM). LightGBM full: F1 0.984
(same as behaviour-only) but MR1 flips 28% and MR4 31%. Contrary to my D11 expectation, the
year effect comes from valid years (LightGBM 38%), not 1980-dated apps (about 0%). MR5/MR6 also
move behaviour-only models (permissions are behaviour), so they measure permission weighting,
not metadata shortcuts. MR3 about 0 everywhere.
Gate 2 reproduction on my MacBook Air (Apple silicon), 21 Nov: `make check` passes (95 tests,
96% coverage); the Dada zip's SHA-256 matches DATA.md; `make rebuild` verified all five
splits; `make pilot` gives the same Gate 2 verdict. `make pilot-compare` verdict: DIFFERENT
(542/750 rows identical, largest rate difference 0.012). LR, LightGBM and MLP are identical,
SVM violation counts identical; only random forests differ, because scikit-learn's trees are not
bit-identical across CPU architectures (same-machine reruns are identical). Decision D15: all
reported numbers come from the reference platform (Linux x86-64); the Mac run is kept in
reports/reproduction/macos-arm64/ as a cross-architecture check.
Step 2.8: SRS and design chapter drafted (30 requirements: 22 functional, 8 non-functional; 23
done with 61 test links, 7 planned and tied to roadmap Checks). The requirements live in
docs/requirements.yaml and tests/test_traceability.py enforces that every one has an existing
verifier (checked by breaking a link on purpose). Performance and usability targets (NFR7,
NFR8) are set now, before Phase 4 measures them. Phase 2 complete.
Step 3.1: one YAML config per experiment (pilot, rq1, rq2, rq3) and one shared engine; the pilot
now runs through it. Check on real data: the pilot config run twice gave identical numbers
(750/750 MR rows, 50/50 detection rows), equal to the committed step 2.7 results. Step 3.2:
PREREGISTRATION.md and the analysis code for H1-H3 committed before any test partition was
opened. H1 keeps the 5% target, adds a per-seed CI rule and 3-of-5 algorithms on both biased
splits (the pilot would give 4 of 5 on DR-AG). H2 pairs 25 algorithm-seed pairs on identical
inputs, with a size-matched arm (a 5-seed Wilcoxon cannot reach p < 0.05). H3 uses MR4 as its
single primary signal; the skew injection and hyperparameter settings are now exact.

## Week 1 (w/c 23 Nov 2026)

-
Step 3.3 (RQ1): `make rq1` ran once (53 min) on the test partitions; prereg-check passed first, each
test partition was opened once (logs/test_access.jsonl), and the analysis code was the pre-registered
version. H1: PARTIALLY SUPPORTED. DR-AG: 4 of 5 algorithms show the shortcut (LR, SVM, RF on MR4;
LightGBM on MR1 and MR4), as in the pilot. VS-AG: only LightGBM (MR1 0.062, MR4 0.069); RF MR4 0.023,
LR/SVM MR4 0.012-0.013, MLP 0.022, all below the pre-registered 5%. Behaviour-only control exactly 0.
VS-AG full and behaviour-only models differ little in F1 (e.g. LR 0.914 vs 0.912), consistent with
little metadata reliance there. Reported as pre-registered; thresholds not changed, no re-run.
Step 3.4 (RQ2): `make rq2` ran once (2 h 7 min); prereg-check passed, each test partition opened
once, analysis code as pre-registered. H2: NOT SUPPORTED, and the effect is reversed: debiased
models violate MORE than biased ones on MR4 (debiased-test inputs: DR 0.243 vs 0.048, VS 0.130 vs
0.046; two-sided p <= 3e-7), also against the size-matched arm and on the biased-test inputs.
Exploratory (not pre-registered, scripts/rq2_exploratory.py): in the DR family the debiased models
flip invalid-year (1980) inputs at 0.20-0.29 vs 0.002 for biased models; LightGBM, LR and SVM
drive the increase. Dada's debiasing does not remove model-level metadata reliance here.
Step 3.5 (RQ3 pool): 300 full-feature models in 60 configurations built from rq3.yaml (34 min, no
test data); every skew level has 3,286 goodware + 3,013 malware; contrary share 0.516 (skew 0) to
0.209 (skew 0.75). Built twice: 300/300 rows identical including validation-probability hashes.
Step 3.6 (RQ3 signals): `make rq3-signals` ran once at 48d7910 (93 min, 2 worker processes).
prereg-check passed; the 2017-19 partition was opened once, for inputs only (labels dropped in
code), at 09:17 UTC, after the pre-registration commit (00:05 UTC): the step 3.2 Check now holds
on a real access. All 10 signals for all 300 models, no missing values. Rerun check: 10 models
recomputed in one process gave identical values. Label-free manipulation check: at skew 0.75
both MR4 and the SHAP metadata share rise for every algorithm. Risk noted before ground truth:
MR4 varies little among un-injected models (per-algorithm means 0.016-0.040).
Step 3.7 (RQ3 ground truth): `make rq3-truth` at 21980c4; a dry run on validation data first
(no test access; every F1 drop exactly 0). Real run: 300 logged load_test() calls for 300 models.
Severe decay: F1 0.87-0.92 on 2012-14 validation to 0.17-0.47 on 2017-19 (mean drop 0.598;
by year 2017 0.710, 2018 0.583, 2019 0.505). Higher injected skew gave slightly LESS decay.
Step 3.8 (H3): `make rq3-analyse`, analysis code identical to the pre-registered version.
Gate 3 outcome: NO SIGNAL. H3a fails: un-injected rho(MR4 pre-deployment, F1 drop) = 0.084
(95% cluster-bootstrap CI -0.297 to 0.446). H3b's rule is met (MR4 - SHAP = 0.516, Holm p
0.003) only because the SHAP share correlates negatively (-0.433); it is not evidence for MR4.
PSI rho 0.813 beats MR4; confidence drop and ATC correlate strongly in the opposite of the
pre-registered direction (-0.78, -0.76). Secondary, descriptive: MR4 measured on 2017-19 inputs
(monitoring) rho 0.760 (0.572 to 0.871); MR1 pre-deployment 0.362. Not computed although listed
as secondary in PREREGISTRATION.md: the plain-Gower realism sensitivity (to disclose).
Roadmap decision: "No signal" -> lead with RQ2 and the tool; discuss framing with the supervisor.
Phase A (21 Nov). A1: froze AndroCT 2015-16 as a locked test-only holdout (7,129 apps; 4,551
goodware, 2,578 malware; not opened; reserved for a later monitoring test). Addendum A
(PREREGISTRATION-ADDENDUM-A.md, f1148b2) committed before running A1-A3:
- A1 (H2 mechanism, training data only): explanation (a) "proxy debiasing leaves the raw
  shortcut" does not hold in either family, because Dada's binned Year/APK size still predict
  the label strongly in our debiased training sets (AUC 0.906 DR, 0.770 VS). Explanation (b)
  "weaker behaviour signal" holds for VS (behaviour AUC 0.923 vs 0.972) and narrowly misses for
  DR (0.970 vs 0.988; margin 0.02).
- A2 (convergent validity): does not hold by its rule. MR4 vs metadata F1 gain rho 0.562,
  95% CI -0.037 to 0.856 (only 10 configurations).
- A3 (plain-Gower sensitivity, promised in the pre-registration): H3 unchanged; un-injected
  rho 0.114 (-0.285 to 0.488) vs 0.084 weighted.
Correction found while interpreting A1: per the Dada DataPort page, Dada's debiasing moves a
dataset's distribution of characteristics toward an AndroZoo extract (representativeness), and
its mixed datasets are balanced per year in training with a year barrier before the test set.
It does not aim to remove every metadata-label link. Our D5 pooled Dada's training and test
parts and re-split them at random, which breaks that protocol. The H2 result therefore holds for
Dada's debiased apps under our re-split protocol, not for Dada used as its authors intended.
Earlier descriptions ("Dada balances metadata between malware and goodware") were too strong.

Addendum B (21-22 Nov). PREREGISTRATION-ADDENDUM-B.md (52260b1) committed before Dada's own test
files were opened (first access 17:52 UTC 21 Nov; 3 load_test() calls, one per test file).
`shortcutcheck addendum b` ran in about 2 h 45 min. Verdict by the pre-registered rule:
**REPLICATES.** VS-AG testable (min 1,209 counted MR4 inputs): MR4 on Dada's debiased test file,
published 0.040 vs debiased-trained 0.143 (Holm p 1.8e-07, Cliff's delta -0.73); size-matched
0.028 vs 0.143 (delta -0.91). DR-AG not testable (min 29 counted MR4 inputs, rule needs 30; its
debiased test file has 40 apps, 4 malware); its tests point the same way but are reported only.
EXPLORATORY caveat (scripts/rq2_dada_exploratory.py): debiased-trained models are also less
stable under the permission MRs that do not touch metadata (MR6 0.051 vs 0.007, 7.0x; MR5 1.4x),
and their F1 is much lower (0.45 vs 0.74-0.79). So the reversal shows generally less stable
models, not specifically more reliance on metadata. Claims must say "less robust to metadata
perturbations", not "more shortcut reliance".
Phase B / Addendum C (22 Nov): metamorphic augmentation fix, proof of concept, training and
validation data only (no test access; test log still 310 lines). PREREGISTRATION-ADDENDUM-C.md
committed (58d48e7) before the code was written; code 1fa4059; run about 5 min (120 models).
A smoke-test dry run (DR-AG_deb, LR, seed 11) printed one model's numbers before the real run;
the rules were already committed. Verdict by the pre-registered rule: **GO (4 of 4 splits)**.
MR4 baseline -> augmented: DR-AG 0.110 -> 0.003; DR-AG_deb 0.234 -> 0.031; VS-AG_deb 0.147 ->
0.021; AndroCT 2012-14 0.023 -> 0.007. F1 drop <= 0.02 everywhere, but VS-AG_deb is on the line
(0.01998; RF and LightGBM there lose 0.03-0.05); GO still holds without it. Held-out MRs: one of
MR1/MR3 falls significantly (Holm) in each split, not both. Reported: contrary-app accuracy rises
in all 4 splits (+0.008 to +0.022; p < 0.01 on DR-AG and AndroCT); permission MRs MR5/MR6
unchanged, so the fix removes metadata dependence specifically and does not fix the general
instability of debiased models (D20). Limits: validation partitions only, 3 seeds, VS-AG
(biased) left out; MR4 falling is partly circular. Next: confirmatory addendum (RQ4) on unseen
data, e.g. the RQ1/RQ2 test partitions and the 2015-16 holdout (decay), before any claim.
An untracked file src/shortcutcheck/training/augment.py (created 01:50, not by this work) was
found in the workspace; it is not used or committed.
Phase C / Addendum D (22 Nov): confirmatory tests. PREREGISTRATION-ADDENDUM-D.md and its code
committed together (4d078ca) before any of its data was opened; dry run on validation data
6.5 min, no test access. Real run about 2 h after Nisula's go; 6 partitions opened once each
(4 RQ1/RQ2 test partitions, 2015-16 holdout for the first time, 2017-19); log now 316 lines.
All 300 pool models matched their step 3.5 hashes before the holdout was opened.
- **H4 SUPPORTED** (3 of 4 splits): MR4 baseline -> fixed on test inputs DR-AG 0.112 -> 0.004,
  DR-AG_deb 0.208 -> 0.025, VS-AG 0.028 -> 0.008, VS-AG_deb 0.146 -> 0.022 (all Holm p 1.2e-07).
  VS-AG_deb fails H4b (F1 0.822 -> 0.799, drop 0.023 > 0.02). Duplicate control: no change in
  MR4, so the effect is the metadata change, not more data. H4d (contrary-app accuracy up):
  3 of 4 splits.
- **H5a HOLDS:** decay 2012-14 -> 2015-16 baseline 0.081 vs fixed 0.054 (p 0.033); duplicate
  control 0.083. Reported: 2017-19 decay 0.626 vs 0.556 (p 2.3e-05).
- **H5b HOLDS:** rho(MR4 at monitoring time on 2015-16 inputs, decay) 0.852 (0.751 to 0.907),
  un-injected 150 models; full pool 0.769. **H5c holds** against PSI (0.196), confidence drop
  (-0.668) and ATC (0.124), Holm p 0.001 each. Pre-deployment MR4 again no signal (-0.090),
  replicating the H3 null on a second period.
EXPLORATORY (scripts/addendum_d_exploratory.py): part of the H5b correlation is between
algorithms; after removing algorithm means rho is 0.521 (un-injected) / 0.727 (full pool), still
above every baseline (PSI 0.153 / 0.044); within each algorithm rho is positive (0.18 to 0.83).
Confidence drop's |rho| is 0.668 but in the opposite direction to its pre-registered
orientation. The fix raised MR5 flips on VS-AG_deb (0.259 -> 0.321).
Addendum E (22 Nov): replication on AZL (Dada's AndroZoo sample, 99 AndroCT apps removed).
PREREGISTRATION-ADDENDUM-E.md, frozen AZL splits and code committed together (461cc75) before
any AZL test data were opened; dry run 36 s, no test access. Run about 9 min after Nisula's go;
AZL 2015-16 and AZL 2017-19 opened once each, AndroCT 2015-16 once more (secondary); log 319.
Verdict by the pre-registered rule: **PARTIALLY REPLICATES**.
- E1 HOLDS: decay to 2015-16 baseline 0.121 vs fixed 0.105 (p 0.013); 2017-19 0.398 vs 0.308.
- E2 DOES NOT HOLD: rho(MR4 monitoring, decay) 0.479 (CI 0.109 to 0.750), below the 0.5 rule.
- E3 HOLDS: algorithm-adjusted rho 0.564 (0.279 to 0.748); AndroCT re-analysis 0.579 (0.322 to
  0.767, secondary).
- E4: beats confidence drop and ATC, not PSI (PSI 0.652 on AZL).
- E5: fix vs behaviour-only: no difference in holdout F1 (0.644 vs 0.650) or decay (0.105 vs
  0.102); same on AndroCT (secondary). Dropping metadata is as good as the fix for accuracy and
  decay. The fix's benefit again comes from LR, SVM and MLP; RF and LightGBM decay slightly more.
Addendum F (22 Nov): mechanism of the fix, training and validation data only (no test access;
log still 319). PREREGISTRATION-ADDENDUM-F.md and code committed (3d81681) before anything was
computed; dry run 9 s; run about 10 min (300 models). Verdict by the pre-registered rule:
**mechanism NOT SUPPORTED**.
- F1 DOES NOT HOLD: metadata-only CV ROC-AUC after augmentation 0.618 (DR-AG) and 0.683
  (DR-AG_deb), above the 0.60 rule; on AndroCT and AZL it falls below 0.5 (0.437, 0.423), i.e.
  the realism filter makes the symmetrisation inexact and can reverse the association.
- F2 and F3 NOT TESTABLE (only 2 splits informative: in-distribution metadata reliance on
  AndroCT and AZL is below 0.005 for linear models). Where testable, linear reliance halves (F2
  pattern) but trees lose MORE reliance than linear models (0.93 vs 0.63), the opposite of F3.
- F4 DOES NOT HOLD (2 of 4 splits).
So the proposed explanation (fix ~ dropping metadata for linear models only) is not supported.
Note for discussion (not tested): in-distribution reliance is tiny on the decay datasets, so the
fix's decay benefit for linear models may come from how metadata (year) extrapolate beyond the
training range rather than from in-distribution reliance; future work.
Phase 4, step 4.1 (22 Nov): release gate `shortcutcheck run MODEL DATA --train TRAIN --mrs core
--threshold 0.05 [--report report.json]` (src/shortcutcheck/gate.py, reused by 4.2 and 4.3).
Exit 0 pass, 1 a relation exceeds the threshold, 2 bad input. Reports each relation's rate with a
Wilson interval, flipped example apps, and MR4 per metadata field. Check passed: a known-bad model
(predicts from metadata) fails with exit 1 and a known-good model (behaviour only) passes with
exit 0 (tests/test_gate.py, 11 tests). Smoke run on DR-AG validation data (no test access): a
LightGBM model fails (MR1 0.287, MR4 0.288; year alone 0.351) in about 5 s. FR20 marked done.
Phase 4, step 4.2 (22 Nov): pytest plugin (src/shortcutcheck/pytest_plugin.py, entry point
pytest11): `@pytest.mark.shortcutcheck(model=..., data=..., train=...)` with the
`shortcutcheck_report` fixture, a direct `shortcutcheck(...)` fixture, and
`--shortcutcheck-reports DIR`. The gate now falls back to built-in Dada defaults when a project
has no configs/phase1.yaml (GateSettings). CI demo in examples/ci-demo (synthetic data; robust
behaviour-only model passes, shortcut-prone LightGBM fails MR1 0.255, MR4 1.000) and workflow
.github/workflows/gate-demo.yml; template for adopters in examples/ci-demo/github-workflow-example.yml.
7 plugin tests. Streamlit 1.67.0 pinned in requirements.txt for step 4.3. FR21 marked done.
