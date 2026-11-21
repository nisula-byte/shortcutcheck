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

