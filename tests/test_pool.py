"""Roadmap step 3.5: the RQ3 model pool, with skew injection, as pre-registered (D18)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from shortcutcheck.cli import app
from shortcutcheck.config import Config, load_config
from shortcutcheck.experiments.compare import compare_tables
from shortcutcheck.experiments.spec import ExperimentConfigError, load_experiment
from shortcutcheck.training.pool import (
    build_pool,
    contrary_mask,
    inject_skew,
    pool_members,
    skew_plan,
)

REPO = Path(__file__).resolve().parents[1]
RQ3 = REPO / "configs" / "experiments" / "rq3.yaml"
runner = CliRunner()


def toy(n: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.uniform(size=n) < 0.45).astype(int))
    x = pd.DataFrame({"apk_size": rng.lognormal(14, 1, size=n), "f": rng.integers(0, 2, n)})
    return x, y


def test_skew_plan_targets_match_the_highest_level() -> None:
    x, y = toy()
    plan = skew_plan(x, y, "apk_size", [0, 0.25, 0.5, 0.75])
    contrary = contrary_mask(x, y, "apk_size", plan.median)
    for c in (0, 1):
        n = int((y == c).sum())
        k = int((contrary & (y == c).to_numpy()).sum())
        assert plan.target[c] == n - int(np.ceil(0.75 * k))


def test_inject_skew_fixes_size_and_raises_skew_with_level() -> None:
    x, y = toy()
    plan = skew_plan(x, y, "apk_size", [0, 0.25, 0.5, 0.75])
    contrary = contrary_mask(x, y, "apk_size", plan.median)
    shares = []
    for level in (0, 0.25, 0.5, 0.75):
        rows = inject_skew(x, y, plan, level, seed=11)
        assert len(set(rows)) == len(rows)
        for c, size in plan.target.items():
            assert int((y.iloc[rows] == c).sum()) == size  # same per-class size at every level
        shares.append(contrary[rows].mean())
    assert shares == sorted(shares, reverse=True) and shares[0] > shares[-1] + 0.1
    # At the top level exactly ceil(0.75 x contrary) are gone in each class.
    top = inject_skew(x, y, plan, 0.75, seed=11)
    for c in (0, 1):
        k = int((contrary & (y == c).to_numpy()).sum())
        kept = int((contrary[top] & (y.iloc[top] == c).to_numpy()).sum())
        assert kept == k - int(np.ceil(0.75 * k))
    a, b = inject_skew(x, y, plan, 0.5, seed=11), inject_skew(x, y, plan, 0.5, seed=11)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, inject_skew(x, y, plan, 0.5, seed=22))


def test_the_configured_pool_has_300_full_feature_models() -> None:
    exp = load_experiment(RQ3, load_config(root=REPO))
    members = pool_members(exp)
    assert len(members) == 300 and len({m.model for m in members}) == 300
    assert len({m.configuration for m in members}) == 60
    zero = [m for m in members if m.skew == 0]
    assert len(zero) == 150 and len({m.configuration for m in zero}) == 30
    assert {m.setting for m in members if m.skew > 0} == {0, 1}


def test_behaviour_only_models_cannot_enter_the_pool(tmp_path: Path) -> None:
    """Roadmap 3.5 Check: behaviour-only models are excluded."""
    raw = yaml.safe_load(RQ3.read_text())
    raw["feature_sets"] = ["full", "behaviour"]
    path = tmp_path / "rq3.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ExperimentConfigError, match="full-feature models only"):
        load_experiment(path, load_config(root=REPO))


def _toy_pool(project: Config) -> Path:
    result = runner.invoke(
        app, ["data", "split-rq3", "--candidate", "SYN-2014-vs-2017", "--name", "rq3-syn"]
    )
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(RQ3.read_text())
    raw["algorithms"] = ["lr", "rf"]
    raw["pool"]["split"] = "rq3-syn"
    raw["pool"]["skew"]["levels"] = [0.0, 0.5]
    raw["pool"]["skew"]["settings_per_level"] = [2, 1]
    raw["pool"]["hyperparameters"] = {
        "lr": [{}, {"C": 0.01}],
        "rf": [{}, {"max_depth": 4}],
    }
    path = project.root / "configs" / "experiments" / "rq3-toy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_pool_build_is_identical_on_rerun_and_opens_no_test_data(project: Config) -> None:
    exp = load_experiment(_toy_pool(project), project)
    first = build_pool(project, exp, "abc", project.root / "m1")
    second = build_pool(project, exp, "abc", project.root / "m2")
    assert len(first) == 2 * 3 * 5  # 2 algorithms x (2 + 1 settings) x 5 seeds
    differ = first.loc[first["val_probability_hash"] != second["val_probability_hash"], "model"]
    assert differ.empty, list(differ)  # validation probabilities bit-identical
    assert compare_tables(first, second).identical
    assert (first["feature_set"] == "full").all()
    assert first.groupby("skew")[["n_goodware", "n_malware"]].nunique().eq(1).all().all()
    assert first.groupby("skew")["contrary_share"].mean().is_monotonic_decreasing
    assert len(list((project.root / "m1").glob("*.joblib"))) == 30
    assert not (project.root / "logs" / "test_access.jsonl").exists()


def test_pool_cli(project: Config) -> None:
    path = _toy_pool(project)
    out = project.root / "out"
    result = runner.invoke(app, ["experiment", "run", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    text = (out / "pool.md").read_text()
    assert "30 models in 6 configurations; feature sets: full" in text
    assert len(pd.read_csv(out / "pool_models.csv")) == 30
