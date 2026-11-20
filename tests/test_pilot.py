"""Step 2.7: Gate 2 criteria evaluated exactly as docs/pilot_plan.md states them."""

from __future__ import annotations

import itertools

import pandas as pd

from shortcutcheck.pilot import PilotResult, gate2

ALGOS = ["lr", "svm", "rf", "lgbm", "mlp"]
SEEDS = [11, 22, 33, 44, 55]


def rows(full_low: dict[str, float], behaviour_violations: int = 0) -> pd.DataFrame:
    """MR1 rows only. `full_low` gives each algorithm's lower CI bound for every seed."""
    out = []
    for a, s in itertools.product(ALGOS, SEEDS):
        out.append(
            {
                "algorithm": a,
                "feature_set": "full",
                "seed": s,
                "mr": "MR1",
                "stratum": "all",
                "n_violations": 10,
                "max_abs_delta": 0.5,
                "rate_low": full_low.get(a, 0.0),
            }
        )
        out.append(
            {
                "algorithm": a,
                "feature_set": "behaviour",
                "seed": s,
                "mr": "MR1",
                "stratum": "all",
                "n_violations": behaviour_violations,
                "max_abs_delta": 0.0,
                "rate_low": 0.0,
            }
        )
    return pd.DataFrame(out)


def result(df: pd.DataFrame) -> PilotResult:
    return PilotResult("toy", "abc", df, pd.DataFrame())


def test_three_clear_algorithms_pass() -> None:
    g = gate2(result(rows({"lr": 0.05, "rf": 0.02, "mlp": 0.011})))
    assert g.behaviour_exact_zero and g.biased_non_zero and g.passes
    assert g.clear_algorithms["rf"] == ["MR1"] and g.clear_algorithms["svm"] == []


def test_two_clear_algorithms_fail() -> None:
    g = gate2(result(rows({"lr": 0.05, "rf": 0.02, "mlp": 0.01})))  # 0.01 is not > 0.01
    assert not g.biased_non_zero and not g.passes


def test_one_bad_seed_disqualifies_an_algorithm() -> None:
    df = rows({"lr": 0.05, "rf": 0.05, "mlp": 0.05})
    df.loc[
        (df["algorithm"] == "mlp") & (df["seed"] == 33) & (df["feature_set"] == "full"), "rate_low"
    ] = 0.004
    assert not gate2(result(df)).passes


def test_any_behaviour_violation_fails_and_is_named() -> None:
    g = gate2(result(rows({a: 0.05 for a in ALGOS}, behaviour_violations=1)))
    assert not g.behaviour_exact_zero and not g.passes
    assert "lr/MR1/s11" in g.behaviour_offenders


def test_secondary_mrs_do_not_count() -> None:
    df = rows({})
    df["mr"] = "MR6"
    df["rate_low"] = 0.5
    assert not gate2(result(df)).biased_non_zero
