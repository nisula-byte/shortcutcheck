"""Addendum A (PREREGISTRATION-ADDENDUM-A.md): decision rules checked on known answers."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shortcutcheck.addenda import AddendumNotCommittedError, phase_a, require_committed
from shortcutcheck.cli import app
from shortcutcheck.config import Config
from shortcutcheck.experiments.spec import load_experiment
from shortcutcheck.training.pool import build_pool
from tests.test_signals import _light_pool

runner = CliRunner()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_require_committed(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    with pytest.raises(AddendumNotCommittedError):
        require_committed(tmp_path, "PLAN.md")
    (tmp_path / "PLAN.md").write_text("rules\n")
    with pytest.raises(AddendumNotCommittedError):
        require_committed(tmp_path, "PLAN.md")  # written but not committed
    _git(tmp_path, "add", "PLAN.md")
    _git(tmp_path, "commit", "-q", "-m", "plan")
    assert len(require_committed(tmp_path, "PLAN.md")) == 40
    (tmp_path / "PLAN.md").write_text("changed rules\n")
    with pytest.raises(AddendumNotCommittedError):
        require_committed(tmp_path, "PLAN.md")


def test_addendum_refuses_to_run_outside_a_commit(project: Config) -> None:
    result = runner.invoke(app, ["addendum", "a2"])
    assert result.exit_code == 2 and "not committed" in result.output


# ---- A1 -------------------------------------------------------------------------------------


def test_cv_auc_known_answers() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=400)})
    y = pd.Series((x["a"] > 0).astype(int))
    assert phase_a.cv_auc(x, y, "lr") > 0.99
    noise = pd.DataFrame({"a": rng.normal(size=400)})
    assert abs(phase_a.cv_auc(noise, y, "lr") - 0.5) < 0.1


def _measures(values: dict[str, dict[tuple[str, str], float]]) -> pd.DataFrame:
    rows = []
    for family, v in values.items():
        for (arm, group), auc in v.items():
            for algorithm in ("lgbm", "lr"):
                rows.append(
                    {
                        "family": family,
                        "set": "s",
                        "arm": arm,
                        "algorithm": algorithm,
                        "group": group,
                        "auc": auc,
                        "sd": 0.01,
                    }
                )
    return pd.DataFrame(rows)


def _family(
    m1_deb: float, m1_bias: float, m2_deb: float, m3_deb: float, m3_bias: float
) -> dict[tuple[str, str], float]:
    return {
        ("debiased", "M1 raw metadata"): m1_deb,
        ("biased, size-matched", "M1 raw metadata"): m1_bias,
        ("debiased", "M2 binned proxies"): m2_deb,
        ("biased, size-matched", "M2 binned proxies"): 0.9,
        ("debiased", "M3 behaviour"): m3_deb,
        ("biased, size-matched", "M3 behaviour"): m3_bias,
    }


def test_judge_a1_rules_and_margins() -> None:
    v = phase_a.judge_a1(
        _measures(
            {
                "DR": _family(0.80, 0.81, 0.55, 0.90, 0.91),  # (a) yes: within margin; (b) no
                "VS": _family(0.70, 0.80, 0.65, 0.80, 0.90),  # (a) no; (b) yes
            }
        )
    )
    dr, vs = v
    assert (dr.family, dr.a_holds, dr.b_holds) == ("DR", True, False)
    assert (vs.family, vs.a_holds, vs.b_holds) == ("VS", False, True)
    # Exactly at the margin still counts; just beyond does not.
    edge = phase_a.judge_a1(
        _measures(
            {
                "DR": _family(0.78, 0.80, 0.60, 0.88, 0.90),
                "VS": _family(0.779, 0.80, 0.601, 0.881, 0.90),
            }
        )
    )
    assert (edge[0].a_holds, edge[0].b_holds) == (True, True)
    assert (edge[1].a_holds, edge[1].b_holds) == (False, False)


# ---- A2 -------------------------------------------------------------------------------------


def _rq1(gain: dict[str, float], noise: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    rows, det = [], []
    for split, algorithm, seed in itertools.product(
        ["rq12-DR-AG", "rq12-VS-AG"], ["lr", "svm", "rf", "lgbm", "mlp"], [11, 22, 33, 44, 55]
    ):
        g = gain[algorithm] * (1 if split.endswith("DR-AG") else 0.5) + rng.normal(0, 0.001)
        rate = rng.uniform(0, 0.4) if noise else g * 4 + rng.normal(0, 0.001)
        base = {"train_split": split, "algorithm": algorithm, "seed": seed}
        for mr in ("MR1", "MR4"):
            rows.append({**base, "feature_set": "full", "mr": mr, "stratum": "all", "rate": rate})
        det.append({**base, "feature_set": "full", "f1": 0.9 + g, "roc_auc": 0.95 + g})
        det.append({**base, "feature_set": "behaviour", "f1": 0.9, "roc_auc": 0.95})
    return pd.DataFrame(rows), pd.DataFrame(det)


GAINS = {"lr": 0.002, "svm": 0.004, "rf": 0.01, "lgbm": 0.05, "mlp": 0.001}


def test_a2_holds_when_violations_track_metadata_gain() -> None:
    rows, det = _rq1(GAINS, noise=False)
    t = phase_a.a2_table(rows, det)
    assert len(t) == 50 and t["cluster"].nunique() == 10
    assert np.allclose(
        t["gain"].groupby(t["algorithm"]).max().sort_index(),
        pd.Series(GAINS).sort_index(),
        atol=0.01,
    )
    r = phase_a.judge_a2(t, resamples=300)
    assert r.rho > 0.9 and r.holds


def test_a2_fails_when_violations_are_unrelated() -> None:
    rows, det = _rq1(GAINS, noise=True)
    assert not phase_a.judge_a2(phase_a.a2_table(rows, det), resamples=300).holds


# ---- A3 -------------------------------------------------------------------------------------


def test_a3_plain_gower_rates_on_a_toy_pool(project: Config) -> None:
    exp = load_experiment(_light_pool(project), project)
    model_dir = project.root / "models"
    build_pool(project, exp, "abc", model_dir)
    plain = phase_a.a3_rates(project, exp, model_dir)
    assert len(plain) == 30 and plain["mr4_pre_plain"].between(0, 1).all()
    rng = np.random.default_rng(0)
    pool = plain[["model"]].assign(
        algorithm=[m.split("-")[0] for m in plain["model"]],
        skew=[float(m.split("-q")[1].split("-")[0]) for m in plain["model"]],
        setting=[int(m.split("-h")[1].split("-")[0]) for m in plain["model"]],
        mr4_pre=rng.uniform(size=30),
        f1_drop=rng.uniform(size=30),
    )
    corr = phase_a.a3_correlations(pool, plain, resamples=50)
    assert set(corr["signal"]) == {"mr4_pre", "mr4_pre_plain"} and len(corr) == 4
