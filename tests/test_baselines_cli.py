"""Step 1.8 and the end-to-end Phase 1 pipeline through the CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shortcutcheck.cli import app
from shortcutcheck.config import Config
from shortcutcheck.metrics import detection_metrics, fpr_at_tpr
from shortcutcheck.training.baselines import ShuffledCheck
from shortcutcheck.training.features import select

runner = CliRunner()


def test_fpr_at_tpr_hand_computed() -> None:
    y = [0, 0, 0, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.8, 0.7, 0.9]
    # Reaching TPR 1.0 needs threshold 0.7, which also admits the 0.8 negative: FPR = 1/4.
    assert fpr_at_tpr(y, s, 0.95) == pytest.approx(0.25)
    m = detection_metrics(y, s)
    assert m["prevalence"] == pytest.approx(1 / 3)
    assert m["roc_auc"] == pytest.approx(7 / 8)


def test_permutation_check_uses_the_mean() -> None:
    # Individual permutations far from chance, but centred on it: no leak.
    wide = ShuffledCheck((0.36, 0.70, 0.45, 0.55, 0.62, 0.40), (0.3,) * 6, 0.3, 0.05)
    assert wide.passes
    # Every permutation well above chance: the signature of leakage.
    leak = ShuffledCheck((0.80, 0.82, 0.79, 0.81, 0.83, 0.80), (0.3,) * 6, 0.3, 0.05)
    assert not leak.passes
    # PR-AUC above prevalence with ROC-AUC at chance is reported but not judged (D7).
    pr_only = ShuffledCheck((0.5,) * 6, (0.6, 0.61, 0.59, 0.6, 0.62, 0.6), 0.3, 0.05)
    assert pr_only.passes and not pr_only.pr_within_margin


def test_pr_auc_is_not_centred_when_one_class_is_spread_out() -> None:
    """Why D7 exists: label-free scores, malware at both extremes, PR-AUC above prevalence."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 5000)
    # Scores never depend on the label, only on how spread out each class is.
    scores = np.where(y == 1, rng.normal(0.5, 0.3, y.size), rng.normal(0.5, 0.05, y.size))
    assert abs(roc_auc_score(y, scores) - 0.5) < 0.02
    assert average_precision_score(y, scores) > 0.55


def test_behaviour_only_drops_metadata(project: Config) -> None:
    frame = pd.DataFrame(
        {"sha256": ["a" * 64], "year": [2014], "apk_size": [10.0], "flag_00": [1], "label": [1]}
    )
    x, _ = select(frame, "behaviour", project.metadata_columns, project.id_column)
    assert list(x.columns) == ["flag_00"]
    x_full, _ = select(frame, "full", project.metadata_columns, project.id_column)
    assert {"year", "apk_size", "flag_00"} <= set(x_full.columns)


def test_dropped_features_never_reach_models(project: Config) -> None:
    frame = pd.DataFrame(
        {
            "sha256": ["a" * 64],
            "year": [2014],
            "dex_date": ["2014-01-02"],
            "flag_00": [1],
            "label": [1],
        }
    )
    assert "dex_date" in project.dropped_features
    assert {"APK size", "Year"} <= set(project.dropped_features)
    for feature_set in ("full", "behaviour"):
        x, _ = select(
            frame,
            feature_set,
            project.metadata_columns,
            project.id_column,
            project.dropped_features,
        )
        assert "dex_date" not in x.columns


def _tree_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_phase1_pipeline_end_to_end(project: Config) -> None:
    before = _tree_digest(project.raw_root)

    result = runner.invoke(app, ["data", "audit"])
    assert result.exit_code == 0, result.output
    assert (project.root / "data_audit.md").exists()

    result = runner.invoke(app, ["data", "validity"])
    assert result.exit_code == 0, result.output
    assert "Overall: KEEP" in result.output  # synthetic data has 15% invalid dex_date

    result = runner.invoke(app, ["data", "split-rq12", "--name", "rq12-syn", "--sets", "SYN-DR-AG"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["data", "split-rq3", "--candidate", "SYN-2014-vs-2017", "--name", "rq3-syn"],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["baselines", "--split", "rq12-syn"])
    assert result.exit_code == 0, result.output
    assert "**Check: PASS.**" in result.output
    table_rows = [ln for ln in result.output.splitlines() if ln.startswith("| rf |")]
    assert table_rows and all(ln.count("|") == 7 for ln in table_rows)
    results = pd.read_csv(project.reports_dir / "rq12-syn_baselines.csv")
    assert len(results) == 2 * 2 * 5  # algorithms x feature sets x seeds
    assert results["mlflow_run_id"].notna().all()

    # The synthetic data plants a year shortcut: full features beat behaviour-only.
    rf = results[results["algorithm"] == "rf"].groupby("feature_set")["roc_auc"].mean()
    assert rf["full"] > rf["behaviour"]

    # Nothing in the pipeline touched the raw data or the test set.
    assert _tree_digest(project.raw_root) == before
    assert not (project.logs_dir / "test_access.jsonl").exists()


def test_baselines_reproducible(project: Config) -> None:
    runner.invoke(app, ["data", "split-rq12", "--name", "r", "--sets", "SYN-DR-AG"])
    runner.invoke(app, ["baselines", "--split", "r"])
    first = pd.read_csv(project.reports_dir / "r_baselines.csv")
    runner.invoke(app, ["baselines", "--split", "r"])
    second = pd.read_csv(project.reports_dir / "r_baselines.csv")
    cols = ["f1", "pr_auc", "roc_auc", "fpr_at_95tpr"]
    np.testing.assert_allclose(first[cols].to_numpy(), second[cols].to_numpy())


def test_mr_applicability_cli(project: Config) -> None:
    """Step 2.2: the applicability report reads only the training partition of a split."""
    runner.invoke(app, ["data", "split-rq12", "--name", "rq12-syn", "--sets", "SYN-DR-AG"])
    result = runner.invoke(app, ["mrs", "applicability", "--split", "rq12-syn"])
    assert result.exit_code == 0, result.output
    text = (project.reports_dir / "mr_applicability.md").read_text()
    assert "| rq12-syn |" in text and "MR1" in text
    assert not (project.root / "logs" / "test_access.jsonl").exists()
