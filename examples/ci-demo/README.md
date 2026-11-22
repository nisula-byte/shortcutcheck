# ShortcutCheck CI demo (roadmap 4.2)

A minimal project that runs the ShortcutCheck release gate as a test, the way a team would add it to
its own repository. The data are synthetic: malware is old and small, goodware new and large, and
the behaviour features carry a real but weaker signal.

| Model | What it learns | Gate |
| --- | --- | --- |
| `robust` | behaviour features only | passes |
| `shortcut` | every feature, so it leans on build year | fails (MR1, MR4) |

## Run it

```bash
pip install -e ../..          # ShortcutCheck, which also installs the pytest plugin
python make_demo.py           # writes data/ and models/
DEMO_MODEL=robust pytest      # 1 passed
DEMO_MODEL=shortcut pytest    # 1 failed: "ShortcutCheck gate failed for MR1, MR4"
```

## The test

```python
@pytest.mark.shortcutcheck(model="models/robust.joblib", data="data/apps.csv",
                           train="data/train.csv", mrs="core", threshold=0.05)
def test_detector_has_no_metadata_shortcuts(shortcutcheck_report):
    shortcutcheck_report.assert_passed()
```

`pytest --shortcutcheck-reports reports` also writes each gate's JSON report, which the dashboard
(`shortcutcheck dashboard reports/<test>.json`) opens.

## In CI

`.github/workflows/gate-demo.yml` in this repository trains the model named in `MODEL` and runs
the gate on every push. With `MODEL` set to `shortcut` the job fails; when the job is a required
status check in branch protection, the pull request cannot be merged.
`github-workflow-example.yml` is a template for your own repository that calls the CLI directly.
