"""Roadmap step 3.6: label-free signals, checked on cases with known answers.

The step's Check: no signal function can receive a label column.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from shortcutcheck import signals as sig
from shortcutcheck.config import Config
from shortcutcheck.data.dada import LABEL
from shortcutcheck.evaluation import rq3_signals as rq3
from shortcutcheck.evaluation.locked_test import access_log
from shortcutcheck.experiments.compare import compare_tables
from shortcutcheck.experiments.spec import load_experiment
from shortcutcheck.training.baselines import build_model
from shortcutcheck.training.pool import build_pool
from tests.test_pool import _toy_pool

LABEL_NAMES = {"y", "y_true", "y_test", "label", "labels", "target", "correct", "correct_test"}


# ---- the Check: no label column can reach a signal ------------------------------------------


def test_assert_unlabelled_rejects_label_columns() -> None:
    ok = pd.DataFrame({"apk_size": [1.0], "year": [2014]})
    assert sig.assert_unlabelled(ok) is ok
    for col in (LABEL, "dataset", "label"):
        with pytest.raises(sig.LabelLeakError):
            sig.assert_unlabelled(ok.assign(**{col: [1]}))


def test_no_signal_function_takes_a_label_argument() -> None:
    """Only the ATC threshold takes correctness, and it is fitted on VALIDATION labels."""
    allowed = {"atc_threshold": {"correct_val"}}
    for name, fn in inspect.getmembers(sig, inspect.isfunction):
        if fn.__module__ != sig.__name__:
            continue
        params = set(inspect.signature(fn).parameters)
        bad = params & (LABEL_NAMES | {"correct_val"}) - allowed.get(name, set())
        assert not bad, (name, bad)
    for fn in (rq3.model_signals, rq3.compute_signals):
        assert not set(inspect.signature(fn).parameters) & LABEL_NAMES


def test_shap_rejects_a_labelled_frame() -> None:
    x, y = _toy(200)
    model = build_model("lr", 1).fit(x, y)
    with pytest.raises(sig.LabelLeakError):
        sig.shap_values(model, "lr", x.assign(**{LABEL: y}), x, seed=1)


# ---- monitoring signals: hand-computed answers ------------------------------------------------


def test_confidence_drop() -> None:
    assert sig.confidence_drop(np.array([0.9, 0.1]), np.array([0.6, 0.5])) == pytest.approx(0.35)


def test_psi_zero_for_equal_and_hand_computed_shift() -> None:
    p = np.linspace(0.005, 0.995, 100)
    assert sig.psi(p, p) == pytest.approx(0.0)
    # All test mass in the top bin: 9 bins hold 0.1 vs floor 1e-4, top bin 0.1 vs 1.0.
    expected = 9 * (1e-4 - 0.1) * np.log(1e-4 / 0.1) + (1.0 - 0.1) * np.log(1.0 / 0.1)
    assert sig.psi(p, np.full(50, 0.999)) == pytest.approx(expected)


def test_psi_merges_duplicate_edges() -> None:
    p_val = np.array([0.0] * 80 + list(np.linspace(0.5, 1, 20)))  # most deciles equal 0
    assert len(sig.psi_edges(p_val)) < 9
    assert sig.psi(p_val, p_val) == pytest.approx(0.0)


def test_atc_threshold_reproduces_validation_accuracy() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 1000)
    correct = rng.uniform(size=1000) < 0.8
    t = sig.atc_threshold(conf, correct)
    assert sig.atc_predicted_accuracy(t, conf) == pytest.approx(correct.mean(), abs=2e-3)
    assert sig.atc_drop(correct.mean(), t, conf) == pytest.approx(0.0, abs=2e-3)
    assert sig.atc_drop(0.9, t, np.full(10, 0.51)) == pytest.approx(0.9)  # all below t


# ---- SHAP metadata share ---------------------------------------------------------------------


def test_metadata_share_hand_computed() -> None:
    values = np.array([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
    # rows: 1/4 and 3/4 (the all-zero row is skipped) -> 0.5
    assert sig.metadata_share(values, ["year", "b", "c"], ["year"]) == pytest.approx(0.5)


def _toy(n: int, shortcut: bool = False, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "apk_size": rng.lognormal(14, 1, n),
            "year": rng.integers(2010, 2015, n).astype(float),
            "f1": rng.integers(0, 2, n).astype(float),
            "f2": rng.integers(0, 2, n).astype(float),
        }
    )
    signal = (x["year"] >= 2012) if shortcut else (x["f1"] + x["f2"] >= 1)
    noise = rng.uniform(size=n) < 0.05
    return x, pd.Series((signal ^ noise).astype(int))


@pytest.mark.parametrize("algorithm", ["lr", "rf", "lgbm", "svm", "mlp"])
def test_shap_share_tracks_metadata_reliance(algorithm: str) -> None:
    """Known answer: a model trained on a year shortcut has a far larger metadata share."""
    shares = {}
    for shortcut in (False, True):
        x, y = _toy(600, shortcut)
        model = build_model(algorithm, 1, n_estimators=30).fit(x, y)
        values = sig.shap_values(model, algorithm, x.iloc[:40], x.iloc[100:130], seed=1)
        assert values.shape == (40, 4)
        shares[shortcut] = sig.metadata_share(values, list(x.columns), ["apk_size", "year"])
    assert shares[True] > 0.6 > 0.4 > shares[False], shares


def test_permutation_shap_is_additive_and_seeded() -> None:
    x, y = _toy(400)
    model = build_model("mlp", 1).fit(x, y)
    xe, xb = x.iloc[:10], x.iloc[200:230]
    v = sig.shap_values(model, "mlp", xe, xb, seed=3)
    assert np.array_equal(v, sig.shap_values(model, "mlp", xe, xb, seed=3))
    base = model.predict_proba(xb)[:, 1].mean()
    assert np.allclose(v.sum(axis=1) + base, model.predict_proba(xe)[:, 1], atol=1e-6)


def test_linear_shap_is_on_the_log_odds() -> None:
    x, y = _toy(400)
    model = Pipeline([("impute", build_model("lr", 1)[0]), ("model", LogisticRegression())])
    model.fit(x, y)
    v = sig.shap_values(model, "lr", x.iloc[:5], x.iloc[100:200], seed=1)
    coef = model[-1].coef_[0]
    centred = x.iloc[:5].to_numpy() - x.iloc[100:200].to_numpy().mean(axis=0)
    assert np.allclose(v, centred * coef)


# ---- end to end on synthetic data ---------------------------------------------------------


def _light_pool(project: Config) -> Path:
    """The toy pool with fast models and small SHAP samples (the real values are in rq3.yaml)."""
    path = _toy_pool(project)
    raw = yaml.safe_load(path.read_text())
    raw["algorithms"] = ["lr", "lgbm"]
    raw["pool"]["hyperparameters"] = {"lr": [{}, {"C": 0.01}], "lgbm": [{}, {"num_leaves": 7}]}
    shap_cfg = raw["pool"]["signals"]["pre_deployment"]["shap_metadata_share"]
    shap_cfg["inputs"], shap_cfg["background"] = 40, 20
    path.write_text(yaml.safe_dump(raw))
    return path


def test_signals_end_to_end_label_free_and_worker_independent(project: Config) -> None:
    exp = load_experiment(_light_pool(project), project)
    model_dir = project.root / "models"
    build_pool(project, exp, "abc", model_dir)
    assert not access_log(project)
    ctx = rq3.build_context(project, exp, model_dir)
    log = access_log(project)
    assert len(log) == 1 and "INPUTS ONLY" in str(log[0]["purpose"])
    assert LABEL not in ctx.x_test.columns
    two = rq3.compute_signals(exp, ctx, workers=2)
    one = rq3.compute_signals(exp, ctx, workers=1)
    assert compare_tables(two, one).identical  # parallelism does not change any value
    applicable = {m.mr_id.lower() for m in ctx.mrs}
    must = [
        c
        for c in rq3.SIGNAL_COLUMNS
        if c[:3] not in {"mr1", "mr3", "cor"} or ({"mr1", "mr3"} <= applicable)
    ]
    assert len(two) == 30 and two[must].notna().all().all()
    assert ((two["mr4_pre"] >= 0) & (two["mr4_pre"] <= 1)).all()
    assert len(access_log(project)) == 1  # nothing else opened the test partition
    with pytest.raises(sig.LabelLeakError):
        rq3.build_context(project, exp, model_dir, x_test=ctx.x_test.assign(**{LABEL: 0}))
