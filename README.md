# ShortcutCheck

Label-free metamorphic testing to detect shortcut learning, and anticipate temporal decay, in
Android malware detectors. Final-year project, Software Engineering (IIT / University of
Westminster). Data: Dada, IEEE DataPort, DOI [10.21227/0frv-zb46](https://doi.org/10.21227/0frv-zb46)
(CC BY 4.0). Feature CSVs only: no APK is ever downloaded or run.

Current state: **Phase 1 done; Phase 2 (MR engine) in progress**: the MR interface, MR1–MR4 and the
runner skeleton are in place. Realism check, metrics with CIs, CLI gate and dashboard come next.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
pre-commit install
pytest                              # roadmap 1.2 Check: green on a second machine or Colab
```

## Phase 1, step by step

Each command writes its evidence file; commit those files. All settings live in
`configs/phase1.yaml`.

| Step | Command | Writes | Check |
| --- | --- | --- | --- |
| 1.3 Get the data | `shortcutcheck data register path/to/dada-dataset-v3.1.zip` | `DATA.md`, read-only `data/raw/dada-v3.1/` | SHA-256 recorded; raw files read-only; every other writer refuses `data/raw` |
| 1.4 Data audit | `shortcutcheck data audit` | `data_audit.md` | Exits non-zero if no RQ3 candidate has ≥1,000 apps per class in both its early and late year windows (counted by each app's own `year`) |
| 1.5 Feature validity | `shortcutcheck data validity` | `reports/dex_date_validity.md` | KEEP or DROP `dex_date` (threshold 20%) |
| 1.6 + 1.7 Dedup and frozen splits | `shortcutcheck data split-rq12 --name <n> --sets <glob>` and `shortcutcheck data split-rq3 --candidate <rq3 candidate>` | `data/splits/<n>/` (commit), `reports/<n>_dedup.md` | Tests assert zero hash overlap and that training code cannot import the test loader |
| 1.8 Baselines | `shortcutcheck baselines --split <n>` | `reports/<n>_baselines.{md,csv}`, MLflow runs in `mlflow.db` | Exits non-zero unless shuffled-label models score at chance |

RQ3 candidates in `configs/phase1.yaml` are named year windows over a family's pooled apps
(e.g. `ACT-G` + `ACT-M`). Dada's `ACT14`/`ACT17` names are time barriers, not app years, so
they are not used to define windows. To rehearse everything on synthetic data first:
`make dry-run`.

## Phase 2: metamorphic relations

A metamorphic relation (MR) is a change to an app's features that should not change the
detector's decision, because it does not change what the app does. The **violation rate** is the
share of inputs whose predicted class flips. It needs no labels.

### The MR interface (`shortcutcheck.mrs.MetamorphicRelation`)

| Member | What it is |
| --- | --- |
| `mr_id`, `name`, `role` | e.g. `"MR1"`, `"Year shift"`, `"core"` or `"secondary"` |
| `justification` | why the change should not matter (domain argument, one or two sentences) |
| `columns_touched` | every column the MR may change; the runner fails if any other column changes |
| `fit(x_train, y_train=None)` | learns the observed range of each touched column from **training** inputs only (MR4 also builds donor pools from training labels) |
| `transform(x, *, rng, predicted=None) -> x'` | a perturbed copy; `x` is never modified; `predicted` is the model's class per row (used by MR4) |
| `is_valid(x') -> bool per row` | are the new values inside the observed training range? |
| `describe()` | the metadata above as a dict, for reports |

`fit` raises `NotApplicableError` with a reason when the data cannot support the MR.

### Core MRs

| MR | Changes | How | Never produces |
| --- | --- | --- | --- |
| MR1 Year shift | `year` | another valid year seen in training, never the app's own | an invalid year (before 2008 or after 2021, e.g. 1980) |
| MR2 SDK shift | `minSdkVersion`, `targetSdkVersion` | another observed (min, target) pair | a pair not seen in training. **Not applicable on Dada**: over 99.9% of values are 0 (missing); see `reports/mr_applicability.md` (D10) |
| MR3 APK size rescale | `apk_size` | × a log-uniform factor in [0.5, 2], clipped | a size outside the training range |
| MR4 Full metadata swap | all metadata columns | copies the metadata of a real training app from the class opposite to the model's prediction | donors with invalid years |

### Running an MR

```python
from shortcutcheck.mrs import YearShift
from shortcutcheck.runner import run_mr

mr = YearShift("year", valid=(2008, 2021)).fit(x_train)
run = run_mr(model, mr, x_val, seed=11)     # no labels needed
run.violation_rate, run.mean_abs_delta, run.n_counted
```

The runner gives the model only its own training columns (`feature_names_in_`), so a
behaviour-only model cannot react to a metadata MR. Statistics count only rows the MR actually
changed **and** whose new values are valid.

| Step | Command | Writes |
| --- | --- | --- |
| 2.2 Which MRs apply | `shortcutcheck mrs applicability` | `reports/mr_applicability.md` |

## Rules the code enforces

- **Test set locked.** Only `shortcutcheck.evaluation.locked_test.load_test()` reads test data.
  Every call is appended to `logs/test_access.jsonl` (commit it). The training loader refuses
  `"test"`, and a test fails if anything outside `shortcutcheck.evaluation` imports the loader.
- **Raw data is read-only.** Registered once, `chmod`-ed read-only, and guarded at write time.
- **Frozen splits** cannot be overwritten; a new split needs a new name.
- **Five seeds**, set in the config; mean ± SD reported.
- **Every number traceable.** Reports carry the git commit; each baseline row carries its
  MLflow run ID.

## Layout

```
configs/phase1.yaml          all Phase 1 settings
src/shortcutcheck/data/      register, audit, validity, dedup, splits, safe writes
src/shortcutcheck/evaluation locked test loader (the only reader of test data)
src/shortcutcheck/training/  feature selection and baselines
src/shortcutcheck/mrs/       MR interface (base.py), MR1-MR4 (core.py), applicability
src/shortcutcheck/runner/    run an MR against a fitted model
src/shortcutcheck/report/    Markdown tables of MR runs
tests/                       pytest suite, runs on synthetic Dada-like data
docs/weekly_log.md           rule 6
```
