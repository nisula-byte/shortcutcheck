"""Roadmap step 3.7: ground truth (F1 drop), with one load_test() call per model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shortcutcheck.analysis import h3
from shortcutcheck.cli import app
from shortcutcheck.config import Config
from shortcutcheck.evaluation import rq3_ground_truth as truth_mod
from shortcutcheck.evaluation import rq3_signals as rq3
from shortcutcheck.evaluation.locked_test import access_log
from shortcutcheck.experiments.spec import load_experiment
from shortcutcheck.training.pool import build_pool
from tests.test_signals import _light_pool

runner = CliRunner()


class Constant:
    feature_names_in_ = np.array(["a"])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile([0.2, 0.8], (len(x), 1))


def test_f1_at_hand_computed() -> None:
    x = pd.DataFrame({"a": [0, 0, 0, 0]})
    # Predicts malware for all: precision 0.5, recall 1 -> F1 2/3.
    assert truth_mod.f1_at(Constant(), x, pd.Series([1, 1, 0, 0]), 0.5) == pytest.approx(2 / 3)
    assert truth_mod.f1_at(Constant(), x, pd.Series([1, 1, 0, 0]), 0.9) == 0.0


def test_one_load_test_per_model_and_pool_table(project: Config) -> None:
    exp = load_experiment(_light_pool(project), project)
    model_dir = project.root / "data" / "models" / "rq3"
    build_pool(project, exp, "abc", model_dir)
    dry = truth_mod.ground_truth(project, exp, model_dir, dry_run=True)
    assert not access_log(project) and len(dry) == 30  # dry run opens nothing

    truth = truth_mod.ground_truth(project, exp, model_dir)
    log = access_log(project)
    assert len(log) == len(truth) == 30  # the Check: one call per model
    named = [next(m for m in truth["model"] if f"model {m}:" in str(e["purpose"])) for e in log]
    assert sorted(named) == sorted(truth["model"])
    assert np.allclose(truth["f1_drop"], truth["f1_val"] - truth["f1_test"])
    assert {"f1_drop_2017", "f1_drop_2018", "f1_drop_2019"} <= set(truth.columns)

    ctx = rq3.build_context(project, exp, model_dir)
    signals = rq3.compute_signals(exp, ctx, workers=1).assign(git_commit="sig")
    pool = truth_mod.assemble_pool(signals, truth, "gt")
    assert set(h3.POOL_COLUMNS) <= set(pool.columns) and len(pool) == 30
    assert (pool["signals_commit"] == "sig").all() and (pool["git_commit"] == "gt").all()
    with pytest.raises(ValueError):
        truth_mod.assemble_pool(signals, truth.drop(columns="f1_drop"), "gt")


def test_ground_truth_cli_dry_run_needs_an_out_folder(project: Config) -> None:
    exp_path = _light_pool(project)
    result = runner.invoke(app, ["experiment", "ground-truth", str(exp_path), "--dry-run"])
    assert result.exit_code == 2 and "--out" in result.output
