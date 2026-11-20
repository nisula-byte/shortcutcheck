# ShortcutCheck

Label-free metamorphic testing to detect shortcut learning, and anticipate temporal decay, in
Android malware detectors. Final-year project, Software Engineering (IIT / University of
Westminster). Data: Dada, IEEE DataPort, DOI [10.21227/0frv-zb46](https://doi.org/10.21227/0frv-zb46)
(CC BY 4.0). Feature CSVs only: no APK is ever downloaded or run.

Current state: **Phase 1 (Foundations)**. MRs, runner, CLI gate and dashboard come in Phases 2–4.

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
src/shortcutcheck/{mrs,runner,report}  Phase 2+
tests/                       pytest suite, runs on synthetic Dada-like data
docs/weekly_log.md           rule 6
```
