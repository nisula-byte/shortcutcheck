"""Step 4.1: the ShortcutCheck gate and `shortcutcheck run`.

Check (roadmap 4.1): a known-bad model (it predicts from metadata) fails the gate with exit code
1; a known-good model (it never sees metadata) passes with exit code 0.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shortcutcheck import gate
from shortcutcheck.cli import app
from shortcutcheck.config import load_config
from shortcutcheck.training.baselines import build_model

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "phase1.yaml"
BEHAVIOUR = [f"api_{i}" for i in range(6)]
METADATA = ["year", "apk_size", "minSdkVersion", "targetSdkVersion"]


def apps(n: int, seed: int, metadata_signal: bool) -> pd.DataFrame:
    """Malware is old and small when ``metadata_signal``; behaviour always carries a weak signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    year = np.where(y == 1, rng.integers(2010, 2013, n), rng.integers(2016, 2019, n))
    if not metadata_signal:
        year = rng.integers(2010, 2019, n)
    size = np.where(y == 1, rng.normal(2e6, 3e5, n), rng.normal(8e6, 1e6, n)).clip(1e5)
    frame = pd.DataFrame(
        {
            "sha256": [f"app{seed}_{i}" for i in range(n)],
            "label": y,
            "year": year,
            "apk_size": size.round(),
            "minSdkVersion": 0,
            "targetSdkVersion": 0,
        }
    )
    for j, col in enumerate(BEHAVIOUR):
        p = np.where(y == 1, 0.55 + 0.05 * j, 0.45 - 0.05 * j)
        frame[col] = (rng.random(n) < p).astype(int)
    return frame


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG, root=REPO)


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory, cfg):
    """A training CSV, an unlabelled input CSV, a known-bad and a known-good model."""
    tmp = tmp_path_factory.mktemp("gate")
    train, data = apps(800, 1, True), apps(300, 2, True)
    train.to_csv(tmp / "train.csv", index=False)
    data.drop(columns=["label"]).to_csv(tmp / "data.csv", index=False)
    x = gate.features(train, cfg)
    bad = build_model("lgbm", 11).fit(x, train["label"])
    good = build_model("lr", 11).fit(x[BEHAVIOUR], train["label"])
    joblib.dump(bad, tmp / "bad.joblib")
    joblib.dump(good, tmp / "good.joblib")
    return tmp


def run(*args: str):
    return CliRunner().invoke(app, ["run", *args, "--config", str(CONFIG)])


def test_known_bad_model_fails_the_gate(files: Path) -> None:
    out = run(
        str(files / "bad.joblib"),
        str(files / "data.csv"),
        "--train",
        str(files / "train.csv"),
        "--report",
        str(files / "bad.json"),
    )
    assert out.exit_code == 1, out.output
    assert "FAIL" in out.output
    report = gate.GateReport.read(files / "bad.json")
    assert not report.passed
    assert "MR4" in report.failing
    mr4 = next(r for r in report.mrs if r.mr == "MR4")
    assert mr4.rate > 0.05 and mr4.ci_low <= mr4.rate <= mr4.ci_high


def test_known_good_model_passes_the_gate(files: Path) -> None:
    out = run(
        str(files / "good.joblib"),
        str(files / "data.csv"),
        "--train",
        str(files / "train.csv"),
        "--report",
        str(files / "good.json"),
    )
    assert out.exit_code == 0, out.output
    report = gate.GateReport.read(files / "good.json")
    assert report.passed
    applicable = [r for r in report.mrs if r.applicable]
    assert applicable and all(r.n_violations == 0 for r in applicable)
    assert report.examples == []


def test_examples_only_change_metadata(files: Path) -> None:
    report = gate.GateReport.read(files / "bad.json")
    assert report.examples
    for e in report.examples:
        assert set(e.changes) <= set(METADATA)
        assert (e.p_before >= 0.5) != (e.p_after >= 0.5)
    assert {f.feature for f in report.feature_rates} <= set(METADATA)


def test_report_round_trip(files: Path, tmp_path: Path) -> None:
    report = gate.GateReport.read(files / "bad.json")
    report.write(tmp_path / "again.json")
    again = gate.GateReport.read(tmp_path / "again.json")
    assert again.to_dict() == report.to_dict()


def test_threshold_one_passes_everything(files: Path) -> None:
    out = run(
        str(files / "bad.joblib"),
        str(files / "data.csv"),
        "--train",
        str(files / "train.csv"),
        "--threshold",
        "1",
    )
    assert out.exit_code == 0, out.output


def test_missing_labels_is_a_usage_error(files: Path) -> None:
    out = run(
        str(files / "bad.joblib"), str(files / "data.csv"), "--train", str(files / "data.csv")
    )
    assert out.exit_code == 2
    assert "label" in out.output


def test_unknown_relation_is_a_usage_error(files: Path) -> None:
    out = run(
        str(files / "bad.joblib"),
        str(files / "data.csv"),
        "--train",
        str(files / "train.csv"),
        "--mrs",
        "MR9",
    )
    assert out.exit_code == 2


def test_parse_mrs() -> None:
    assert gate.parse_mrs("core") == ("MR1", "MR3", "MR4")
    assert gate.parse_mrs("ALL") == ("MR1", "MR3", "MR4", "MR5", "MR6")
    assert gate.parse_mrs("mr4, MR1,MR4") == ("MR4", "MR1")
    with pytest.raises(gate.GateError):
        gate.parse_mrs("")


def test_inapplicable_relation_does_not_fail(cfg) -> None:
    train = apps(400, 3, True).drop(columns=["year"])
    x = gate.features(train, cfg)
    model = build_model("lr", 11).fit(x[BEHAVIOUR], train["label"])
    report = gate.check(model, x, train["label"], x, cfg, mrs=("MR1",), realism=False)
    assert report.mrs[0].applicable is False
    assert report.passed


def test_bad_threshold_rejected(cfg) -> None:
    train = apps(200, 4, True)
    x = gate.features(train, cfg)
    with pytest.raises(gate.GateError):
        gate.check(None, x, train["label"], x, cfg, threshold=1.5)


def test_sampling_is_reproducible(files: Path, cfg) -> None:
    model = joblib.load(files / "bad.joblib")
    train = pd.read_csv(files / "train.csv")
    data = pd.read_csv(files / "data.csv")
    kw = dict(mrs=("MR4",), sample=100, seed=7)
    a = gate.check(
        model, gate.features(train, cfg), train["label"], gate.features(data, cfg), cfg, **kw
    )
    b = gate.check(
        model, gate.features(train, cfg), train["label"], gate.features(data, cfg), cfg, **kw
    )
    assert a.n_checked == 100
    assert a.mrs[0].rate == b.mrs[0].rate
