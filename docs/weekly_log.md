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

