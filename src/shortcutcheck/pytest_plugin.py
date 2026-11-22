"""pytest plugin (roadmap 4.2): run the ShortcutCheck gate inside a project's test suite.

Installed automatically with the package (entry point ``pytest11``). Two ways to use it:

    import pytest

    @pytest.mark.shortcutcheck(model="models/detector.joblib", data="data/apps.csv",
                               train="data/train.csv", mrs="core", threshold=0.05)
    def test_detector_has_no_metadata_shortcuts(shortcutcheck_report):
        assert shortcutcheck_report.passed, shortcutcheck_report.render()

or call the gate directly with the ``shortcutcheck`` fixture:

    def test_detector(shortcutcheck):
        shortcutcheck("models/detector.joblib", "data/apps.csv", "data/train.csv").assert_passed()

Paths are relative to the pytest root directory. ``--shortcutcheck-reports DIR`` writes each
test's JSON report (for the dashboard).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shortcutcheck import gate

MARKER = "shortcutcheck"
_OPTIONS = ("mrs", "threshold", "sample", "seed", "realism")


class ShortcutCheckFailed(AssertionError):
    """Raised by ``GateResult.assert_passed`` with the rendered report."""


class GateResult:
    """A gate report bound to a helpful assertion."""

    def __init__(self, report: gate.GateReport) -> None:
        self.report = report

    @property
    def passed(self) -> bool:
        return self.report.passed

    def render(self) -> str:
        return self.report.render()

    def assert_passed(self) -> None:
        if not self.report.passed:
            raise ShortcutCheckFailed(
                f"ShortcutCheck gate failed for {', '.join(self.report.failing)}\n\n"
                + self.report.render()
            )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shortcutcheck")
    group.addoption(
        "--shortcutcheck-reports",
        default=None,
        metavar="DIR",
        help="write the JSON report of every ShortcutCheck gate run to DIR",
    )
    group.addoption(
        "--shortcutcheck-config",
        default=None,
        metavar="FILE",
        help="project config with metadata and dropped columns (default: built-in Dada defaults)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}(model, data, train, mrs='core', threshold=0.05, sample=5000, seed=11, "
        "realism=True): run the ShortcutCheck metamorphic gate; the result is available as the "
        "shortcutcheck_report fixture",
    )


def _run(
    config: pytest.Config, name: str, model: str, data: str, train: str, **kw: Any
) -> GateResult:
    root = Path(str(config.rootpath))
    cfg_path = config.getoption("--shortcutcheck-config")
    settings = gate.resolve_settings(Path(cfg_path) if cfg_path else None)
    if "mrs" in kw and isinstance(kw["mrs"], str):
        kw["mrs"] = gate.parse_mrs(kw["mrs"])
    report = gate.check_files(root / model, root / data, root / train, settings, **kw)
    out = config.getoption("--shortcutcheck-reports")
    if out:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        report.write(root / out / f"{safe}.json")
    return GateResult(report)


@pytest.fixture
def shortcutcheck(request: pytest.FixtureRequest) -> Callable[..., GateResult]:
    """Run the gate: ``shortcutcheck(model, data, train, mrs=..., threshold=...)``."""

    def call(model: str, data: str, train: str, **kw: Any) -> GateResult:
        unknown = set(kw) - set(_OPTIONS)
        if unknown:
            raise TypeError(f"unknown ShortcutCheck option(s): {sorted(unknown)}")
        return _run(request.config, request.node.nodeid, model, data, train, **kw)

    return call


@pytest.fixture
def shortcutcheck_report(request: pytest.FixtureRequest) -> GateResult:
    """The gate result for the test's ``@pytest.mark.shortcutcheck(...)`` marker."""
    marker = request.node.get_closest_marker(MARKER)
    if marker is None:
        raise pytest.UsageError("shortcutcheck_report needs @pytest.mark.shortcutcheck(...)")
    kw = dict(marker.kwargs)
    missing = [k for k in ("model", "data", "train") if k not in kw]
    if missing:
        raise pytest.UsageError(f"@pytest.mark.shortcutcheck is missing {missing}")
    model, data, train = kw.pop("model"), kw.pop("data"), kw.pop("train")
    unknown = set(kw) - set(_OPTIONS)
    if unknown:
        raise pytest.UsageError(f"unknown ShortcutCheck option(s): {sorted(unknown)}")
    return _run(request.config, request.node.nodeid, model, data, train, **kw)
