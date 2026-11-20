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

## Week 1 (w/c 23 Nov 2026)

-
