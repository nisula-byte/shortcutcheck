"""Step 4.3: the dashboard loads reports produced by the CLI, not hard-coded data."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from shortcutcheck import dashboard, gate
from shortcutcheck.cli import app

APP = str(Path(dashboard.__file__).with_name("dashboard_app.py"))
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml"


@pytest.fixture(scope="module")
def reports(files: Path) -> dict[str, Path]:
    """Reports written by the real CLI for the known-bad and known-good models."""
    out = {}
    for name in ("bad", "good"):
        path = files / f"dash_{name}.json"
        CliRunner().invoke(
            app,
            [
                "run",
                str(files / f"{name}.joblib"),
                str(files / "data.csv"),
                "--train",
                str(files / "train.csv"),
                "--report",
                str(path),
                "--config",
                str(CONFIG),
            ],
        )
        assert path.exists()
        out[name] = path
    return out


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.test_gate import files as make_files  # reuse the 4.1 fixture's data and models

    return make_files.__wrapped__(tmp_path_factory, gate.resolve_settings(CONFIG))  # type: ignore[attr-defined]


def run_app(path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv(dashboard.ENV, str(path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_failing_report_shows_fail_and_drill_down(reports, monkeypatch) -> None:
    at = run_app(reports["bad"], monkeypatch)
    assert "FAIL" in at.error[0].value and "MR4" in at.error[0].value
    report = gate.GateReport.read(reports["bad"])
    shown = at.dataframe[0].value
    assert list(shown["relation"]) == [r.mr for r in report.mrs]
    assert shown["flips"].tolist() == [r.n_violations for r in report.mrs]
    # Drill-down: choose MR4, then one flipped app; its changed fields are listed.
    at.selectbox[0].select("MR4").run()
    examples = at.dataframe[-2].value
    assert len(examples) == sum(e.mr == "MR4" for e in report.examples)
    changes = at.dataframe[-1].value
    assert set(changes["field"]) <= {"year", "apk_size", "minSdkVersion", "targetSdkVersion"}


def test_passing_report_shows_pass(reports, monkeypatch) -> None:
    at = run_app(reports["good"], monkeypatch)
    assert at.success and "PASS" in at.success[0].value
    assert not at.error


def test_without_a_report_it_asks_for_one(monkeypatch) -> None:
    monkeypatch.delenv(dashboard.ENV, raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.info and "No report loaded" in at.info[0].value


def test_broken_report_is_reported(tmp_path: Path, monkeypatch) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    at = run_app(bad, monkeypatch)
    assert "Could not read the report" in at.error[0].value


def test_frames_follow_the_report(reports) -> None:
    report = gate.GateReport.read(reports["bad"])
    assert len(dashboard.mr_frame(report)) == len(report.mrs)
    assert set(dashboard.feature_frame(report)["field"]) <= {
        f.feature for f in report.feature_rates
    }
    assert dashboard.report_path(["x.json"]) == Path("x.json")
