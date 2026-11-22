"""Step 4.2: the pytest plugin. A shortcut-prone model fails the test; a robust one passes."""

from __future__ import annotations

import json

import joblib
import pytest

from shortcutcheck import gate
from shortcutcheck.training.baselines import build_model
from tests.test_gate import BEHAVIOUR, apps

TEST_FILE = """
import pytest

@pytest.mark.shortcutcheck(model="{model}.joblib", data="apps.csv", train="train.csv",
                           mrs="core", threshold=0.05)
def test_gate(shortcutcheck_report):
    shortcutcheck_report.assert_passed()
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    train, data = apps(600, 1, True), apps(250, 2, True)
    train.to_csv(pytester.path / "train.csv", index=False)
    data.drop(columns=["label"]).to_csv(pytester.path / "apps.csv", index=False)
    x = gate.features(train, gate.GateSettings())
    joblib.dump(build_model("lgbm", 11).fit(x, train["label"]), pytester.path / "bad.joblib")
    joblib.dump(
        build_model("lr", 11).fit(x[BEHAVIOUR], train["label"]), pytester.path / "good.joblib"
    )
    return pytester


def test_marker_passes_a_robust_model(project: pytest.Pytester) -> None:
    project.makepyfile(TEST_FILE.format(model="good"))
    project.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)


def test_marker_fails_a_shortcut_prone_model(project: pytest.Pytester) -> None:
    project.makepyfile(TEST_FILE.format(model="bad"))
    result = project.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*ShortcutCheck gate failed for*MR4*"])


def test_reports_option_writes_json(project: pytest.Pytester) -> None:
    project.makepyfile(TEST_FILE.format(model="bad"))
    project.runpytest_subprocess("-p", "no:cacheprovider", "--shortcutcheck-reports", "out")
    files = list((project.path / "out").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["passed"] is False
    assert gate.GateReport.read(files[0]).failing


def test_fixture_can_be_called_directly(project: pytest.Pytester) -> None:
    project.makepyfile("""
        def test_direct(shortcutcheck):
            result = shortcutcheck("good.joblib", "apps.csv", "train.csv", mrs="MR4")
            assert result.passed
            assert [r.mr for r in result.report.mrs] == ["MR4"]
        """)
    project.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)


def test_marker_without_paths_is_a_usage_error(project: pytest.Pytester) -> None:
    project.makepyfile("""
        import pytest

        @pytest.mark.shortcutcheck(model="good.joblib")
        def test_gate(shortcutcheck_report):
            pass
        """)
    result = project.runpytest_subprocess("-p", "no:cacheprovider")
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*missing*data*train*"])


def test_unknown_option_is_rejected(project: pytest.Pytester) -> None:
    project.makepyfile("""
        def test_direct(shortcutcheck):
            shortcutcheck("good.joblib", "apps.csv", "train.csv", treshold=0.1)
        """)
    project.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(failed=1)


def test_marker_is_registered(project: pytest.Pytester) -> None:
    result = project.runpytest_subprocess("--markers")
    result.stdout.fnmatch_lines(["*@pytest.mark.shortcutcheck(model, data, train*"])
