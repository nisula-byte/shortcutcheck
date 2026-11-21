# Pre-registration addendum B: does the H2 reversal hold under Dada's own protocol?

Nisula Dasanayake · 21 November 2026

This addendum is committed **before** the experiment below runs and before any of its test
data is opened. It does not change H1–H3 or their verdicts. It tests whether the H2 result
(debiased models violate MR4 *more* than biased ones; `reports/rq2/h2.md`) depends on how the
data was split.

## Why

H2 used splits built under D5: Dada's training and test files were pooled and re-split at
random. Dada's own protocol, per its DataPort page, is different. The training files are
balanced between goodware and malware per year, a year barrier separates training from test,
and debiasing moves the dataset toward an AndroZoo extract. D5 breaks that protocol (D19).
Addendum A (A1) found that Dada's binned Year and APK size still predict the label in our
re-split debiased training sets. So the H2 reversal may be an artefact of D5. This addendum
tests the same question under Dada's protocol.

**Known before writing this:** every H1–H3 and addendum A result, and the size and class counts
of the new splits (from their manifests): `dada-DR-AG` train 10,608 (Dada ships no DR-AG test
file); `dada-DR-AG_deb` 1,614 train / **40 test (4 malware)**; `dada-VS-AG` 133,238 / 6,113;
`dada-VS-AG_deb` 3,492 / 1,333 (frozen in 35a02a3, not opened). The DR test file is therefore
tiny, and the rules below make **VS the deciding family** for that reason. The decision was
made before any result exists.

## Design (`configs/experiments/rq2-dada.yaml`)

The same as RQ2 (`configs/experiments/rq2.yaml`, D17) except for the splits:

- **Models:** full features; LR, SVM, RF, LightGBM, MLP; seeds 11, 22, 33, 44, 55; untuned.
  Three arms per family, matched by algorithm and seed:
  - **published:** trained on Dada's biased training file;
  - **size-matched:** that file cut (stratified, per seed) to the debiased training file's size;
  - **debiased:** trained on Dada's debiased training file.
- **Inputs:**
  - `h2-deb` (**primary**): Dada's debiased test file (after the year barrier), with every app
    that any arm of the family trained on removed;
  - `h2-biased` (secondary, VS only, because Dada ships no DR-AG test file): Dada's biased test
    file, with the same removal.
- **Everything else as RQ2:** MRs and the realism check fitted on the evaluation split's
  training part; uniform 5,000-app MR sample (seed 11) for larger inputs; each test file opened
  once through `load_test()`.

## Tests

For each family, input set and core MR (MR1, MR3, MR4): d = rate(biased) − rate(debiased) over
the 25 pairs. Two one-sided Wilcoxon signed-rank tests (SciPy, `zero_method="wilcox"`,
`method="auto"`; all-zero d gives p = 1):

- **reversal** (d < 0, biased below debiased);
- **original H2 direction** (d > 0).

Holm runs across the core-MR tests of both families, separately for each direction, comparison
(published or size-matched) and input set. Cliff's delta is reported next to every p-value. The
original H2 code (`analysis/h2.py`, unchanged) is also applied to these results for
comparability.

**Testability:** a family is testable on MR4 only if every one of its 50 published and debiased
models has **at least 30 counted MR4 inputs** on `h2-deb`. Otherwise its results are reported
without a verdict.

## Decision rules (α = 0.05, MR4, published vs debiased, `h2-deb`)

Per testable family:

- **reversal replicates:** the reversal test's Holm p < 0.05;
- **original direction:** the original-direction test's Holm p < 0.05;
- **no difference:** neither.

Overall:

- **"The reversal replicates under Dada's protocol"** if VS is testable and its reversal
  replicates, and DR either replicates or is not testable.
- **"Partially"** if exactly one testable family replicates.
- **"Does not replicate"** if no testable family replicates. In that case, report which
  direction, if any, holds.
- **"Not testable"** if VS is not testable.

**Robust to training size:** per family, the same reversal test with the size-matched arm in
place of the published arm also has Holm p < 0.05.

## How each outcome will be reported

- **Replicates:** H2's reversal stands under both protocols. The paper can state that models
  trained on Dada's debiased datasets rely more on metadata than models trained on the biased
  ones, under Dada's own protocol.
- **Does not replicate:** the H2 reversal is specific to our re-split (D5). The paper reports
  that the conclusion depends on the evaluation protocol, a methodological finding in its own
  right.
- **Partially or not testable:** reported as such, with the DR power limitation stated.

The H2 verdict itself (NOT SUPPORTED, judged under D5 as pre-registered) is not changed by any of
these outcomes.
