# Reproducing the results on a second machine

Use this guide to check, on a machine other than the one that produced them, that the code passes
its checks and that the committed results come out the same. For Gate 2 this is the pilot RQ1
(roadmap step 2.7). Written for macOS on Apple silicon; Linux works the same way without the
Homebrew steps.

Time needed: about 30 minutes, most of it downloading and installing.

## 1. Install the tools (once)

Open **Terminal** and run:

```bash
# Homebrew (skip if `brew --version` already works); follow its instructions at the end
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11 libomp git
python3.11 --version        # must print Python 3.11.x
```

`libomp` is needed by LightGBM on macOS. Without it, importing LightGBM fails with a
`libomp.dylib` error.

## 2. Get the code

If you already have the repository:

```bash
cd ~/path/to/shortcutcheck
git checkout main
git pull
```

Otherwise, clone it: `git clone https://github.com/nisula-byte/shortcutcheck.git && cd shortcutcheck`.

## 3. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # do this again in every new Terminal window
python -m pip install --upgrade pip
make setup                         # installs the pinned packages and the git hooks
```

## 4. Check the code (`make check`)

```bash
make check
```

This runs the linters (ruff, black), strict type checking (mypy) and the full test suite on
synthetic data. **Expected:** it ends without errors and every test passes. If anything fails,
stop and keep the full output.

## 5. Add the Dada data

The raw data is not in Git (Dada is cited, not re-hosted). Download
`dada-dataset-v3_1.zip` from IEEE DataPort (DOI 10.21227/0frv-zb46; free IEEE account), then:

```bash
shortcutcheck data register ~/Downloads/dada-dataset-v3_1.zip
```

**Expected:** the zip's SHA-256 matches the one recorded in `DATA.md`, so no new row is added
and `git status` shows `DATA.md` unchanged. If the checksums differ, the command stops: it is
not the same data.

## 6. Rebuild the processed data (`make rebuild`)

```bash
make rebuild
```

Each frozen split is recomputed from the raw data with the same code and seed, then checked
against its committed index files and manifest before anything is written. **Expected:** five
lines saying `Rebuilt '<split>': matches its frozen index files and manifest.` The test sets are
rebuilt but never read.

## 7. Run the pilot and compare (`make pilot`, `make pilot-compare`)

```bash
make pilot            # about 2 minutes; overwrites reports/pilot_rq1_rq12-DR-AG.*
make pilot-compare    # compares your run with the committed results
```

`make pilot-compare` prints one of three verdicts:

| Verdict | Meaning | Gate 2 reproduction |
| --- | --- | --- |
| `IDENTICAL` | every count and rate equal | passes |
| `MATCHES` | every rate within 0.01, same Gate 2 verdict | passes; note the largest difference |
| `DIFFERENT` | larger differences, or a different Gate 2 verdict | investigate (see below) |

What to expect on a different processor type (decision D15). On the same machine, two runs
are identical. Across processor architectures (Apple silicon arm64 vs Intel/AMD x86-64), the
first reproduction (`reports/reproduction/macos-arm64/`) found:

- logistic regression, LightGBM and the MLP are **identical**;
- the linear SVM differs in the 4th decimal place of its accuracy, with identical violation
  counts;
- **random forests are trained slightly differently**: scikit-learn's tree building is not
  bit-identical across processor types. Violation rates moved by up to 0.009 (0.012 within a
  subgroup), so the verdict was `DIFFERENT` under the 0.01 rule, with the same Gate 2 verdict.

So on another architecture, a `DIFFERENT` verdict that comes only from random forest rows, with
the same Gate 2 verdict, repeats the known cross-architecture effect. Check the per-algorithm
table:

```bash
shortcutcheck pilot-reproduction reports/reproduction/<machine> --label "<machine description>"
```

after copying your two pilot CSVs into `reports/reproduction/<machine>/`. All reported numbers
come from the reference platform (Linux x86-64).

## 8. Record the result and clean up

1. Write the verdict, the largest difference and your machine (e.g. "MacBook Air M2, macOS 15")
   in `docs/weekly_log.md`.
2. Put the generated report back, so the committed results stay the reference:

   ```bash
   git checkout -- reports/
   git status            # should show only your weekly_log.md change
   ```

3. Commit the log entry on a branch and open a pull request.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Library not loaded: libomp.dylib` | `brew install libomp` |
| `python3.11: command not found` | `brew install python@3.11`, then open a new Terminal |
| `shortcutcheck: command not found` | run `source .venv/bin/activate` in this Terminal |
| `data/raw/dada-v3.1 already holds data` | the data is already registered; go to step 6 |
| `Split '...': apps differ from the frozen index file` | different raw data or code: check the SHA-256 in step 5 and that you are on `main` |
