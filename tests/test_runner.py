"""Runner and report (roadmap step 2.1), with early oracle models ahead of step 2.5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from shortcutcheck.mrs import MetamorphicRelation, YearShift, core_mrs
from shortcutcheck.report import render_runs
from shortcutcheck.runner import ContractViolation, MRRun, run_mr

VALID = (2008, 2021)


def apps(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    x = pd.DataFrame(
        {
            "year": np.where(y == 1, rng.integers(2012, 2015, n), rng.integers(2016, 2020, n)),
            "minSdkVersion": 0,
            "targetSdkVersion": 0,
            "apk_size": rng.integers(10_000, 50_000_000, n),
            "send_sms": rng.integers(0, 2, n),
            "api_calls": rng.integers(0, 300, n),
        }
    )
    return x, y


class YearOnly:
    """Predicts malware for apps built before 2015: a pure year shortcut."""

    feature_names_in_ = np.array(["year"])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = (x["year"].to_numpy() < 2015).astype(float)
        return np.column_stack([1 - p, p])


class Constant:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile([0.2, 0.8], (len(x), 1))


def test_year_only_model_violates_mr1_and_behaviour_only_model_does_not() -> None:
    x, y = apps()
    mr1 = YearShift("year", VALID).fit(x)
    shortcut = run_mr(YearOnly(), mr1, x, seed=11)
    assert shortcut.violation_rate > 0.3

    behaviour = RandomForestClassifier(n_estimators=20, random_state=0)
    behaviour.fit(x[["send_sms", "api_calls"]], y)
    for mr in core_mrs(list(x.columns), VALID):
        try:
            mr.fit(x, y)
        except Exception:  # MR2 is not applicable here (all SDK values are 0)
            continue
        run = run_mr(behaviour, mr, x, seed=11)
        assert run.n_violations == 0 and run.mean_abs_delta == 0.0, mr.mr_id
        assert run.n_counted > 0


def test_constant_model_never_flips() -> None:
    x, y = apps()
    run = run_mr(Constant(), YearShift("year", VALID).fit(x), x, seed=1)
    assert run.violation_rate == 0.0


class Leaky(MetamorphicRelation):
    mr_id = "BAD"
    name = "changes an undeclared column"
    justification = "test double"

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        x["api_calls"] = x["api_calls"] + 1
        return x


def test_runner_rejects_an_mr_that_breaks_its_contract() -> None:
    x, _ = apps(20)
    with pytest.raises(ContractViolation, match="api_calls"):
        run_mr(Constant(), Leaky(("year",)).fit(x), x, seed=0)


def test_statistics_on_a_hand_computed_example() -> None:
    run = MRRun(
        mr_id="MR1",
        p_before=np.array([0.9, 0.9, 0.2, 0.2, 0.6]),
        p_after=np.array([0.1, 0.8, 0.7, 0.2, 0.4]),
        perturbed=np.array([True, True, True, False, True]),
        valid=np.array([True, True, True, True, False]),
    )
    # Counted rows: 0, 1, 2. Flips among them: 0 and 2.
    assert run.n_counted == 3 and run.n_violations == 2
    assert run.violation_rate == pytest.approx(2 / 3)
    assert run.mean_abs_delta == pytest.approx((0.8 + 0.1 + 0.5) / 3)
    empty = MRRun("MR1", np.zeros(2), np.zeros(2), np.zeros(2, bool), np.ones(2, bool))
    assert np.isnan(empty.violation_rate) and np.isnan(empty.mean_abs_delta)
    table = render_runs([run, empty], "Demo")
    assert "| MR1 | 0 | 5 | 4 | 1 | 3 | 2 | 0.667 | 0.467 |" in table
    assert "| – |" in table
