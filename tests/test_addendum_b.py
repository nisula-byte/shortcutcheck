"""Addendum B decision rules, on hand-built results with known answers."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from shortcutcheck.addenda import phase_b

ALGOS = ["lr", "svm", "rf", "lgbm", "mlp"]
SEEDS = [11, 22, 33, 44, 55]


def rows(shift: dict[str, float], counted: dict[str, int]) -> pd.DataFrame:
    """MR4 rate: debiased = base; published and size-matched = base - shift[family]."""
    rng = np.random.default_rng(3)
    out = []
    for family, (a, s) in itertools.product(phase_b.FAMILIES, itertools.product(ALGOS, SEEDS)):
        base = 0.10 + rng.uniform(0, 0.02)
        for run, rate in (
            (family, base - shift[family]),
            (f"{family}-size-matched", base - shift[family]),
            (f"{family}_deb", base),
        ):
            for mr in ("MR1", "MR4"):
                out.append(
                    {
                        "run": run,
                        "role": "h2-deb",
                        "algorithm": a,
                        "seed": s,
                        "mr": mr,
                        "feature_set": "full",
                        "stratum": "all",
                        "rate": rate if mr == "MR4" else base,
                        "n_counted": counted[family],
                    }
                )
    return pd.DataFrame(out)


@pytest.mark.parametrize(
    ("shift", "counted", "verdict"),
    [
        ({"DR-AG": 0.05, "VS-AG": 0.05}, {"DR-AG": 200, "VS-AG": 900}, "REPLICATES"),
        ({"DR-AG": 0.0, "VS-AG": 0.05}, {"DR-AG": 20, "VS-AG": 900}, "REPLICATES"),
        ({"DR-AG": 0.0, "VS-AG": 0.05}, {"DR-AG": 200, "VS-AG": 900}, "PARTIALLY REPLICATES"),
        ({"DR-AG": 0.05, "VS-AG": 0.0}, {"DR-AG": 200, "VS-AG": 900}, "PARTIALLY REPLICATES"),
        ({"DR-AG": 0.05, "VS-AG": -0.05}, {"DR-AG": 20, "VS-AG": 900}, "DOES NOT REPLICATE"),
        ({"DR-AG": 0.05, "VS-AG": 0.05}, {"DR-AG": 200, "VS-AG": 29}, "NOT TESTABLE"),
    ],
)
def test_overall_verdicts(shift: dict, counted: dict, verdict: str) -> None:
    assert phase_b.judge(rows(shift, counted)).verdict == verdict


def test_family_outcomes_directions_and_robustness() -> None:
    r = phase_b.judge(rows({"DR-AG": 0.0, "VS-AG": -0.05}, {"DR-AG": 200, "VS-AG": 900}))
    by = {f.family: f for f in r.families}
    assert by["VS-AG"].outcome == "original direction" and not by["VS-AG"].robust
    assert by["DR-AG"].outcome == "no difference"
    r = phase_b.judge(rows({"DR-AG": 0.05, "VS-AG": 0.05}, {"DR-AG": 200, "VS-AG": 900}))
    assert all(f.robust for f in r.families)
    mr4 = r.table[(r.table["mr"] == "MR4") & (r.table["direction"] == "reversal")]
    assert (mr4["cliffs_delta"] == -1.0).all()  # every biased rate below every debiased rate


def test_testability_threshold_is_inclusive() -> None:
    ok, least = phase_b.testable(
        rows({"DR-AG": 0, "VS-AG": 0}, {"DR-AG": 30, "VS-AG": 30}), "DR-AG"
    )
    assert ok and least == 30
