"""Addendum F: reliance, agreement and the decision rules on known answers."""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from shortcutcheck.addenda import mechanism as m


def toy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    x = pd.DataFrame({"year": y * 3 + rng.normal(0, 0.5, n), "perm": y + rng.normal(0, 1.5, n)})
    return x, y


def test_reliance_is_large_for_a_metadata_model_and_zero_for_a_behaviour_model() -> None:
    x, y = toy()
    meta_model = LogisticRegression().fit(x, y)
    beh_model = LogisticRegression().fit(x[["perm"]], y)
    _, r_meta = m.reliance(meta_model, x, y, ["year"])
    _, r_beh = m.reliance(beh_model, x, y, ["year"])
    assert r_meta > 0.1
    assert r_beh == pytest.approx(0.0)
    assert m.agreement(beh_model, beh_model, x) == 1.0


def rows(lin_aug: float, tree_aug: float, agree_gain: float) -> pd.DataFrame:
    out = []
    rng = np.random.default_rng(3)
    for split, alg, seed in itertools.product("wxyz", m.ALGORITHMS, m.SEEDS):
        base_r = 0.10
        aug_r = base_r * (lin_aug if alg in m.LINEAR else tree_aug)
        base_ag = 0.90 + rng.uniform(0, 0.005)
        for arm, r, ag in (
            ("baseline", base_r, base_ag),
            ("augmented", aug_r, base_ag + agree_gain + rng.uniform(-0.001, 0.001)),
            ("behaviour", 0.0, 1.0),
        ):
            out.append(
                {
                    "split": split,
                    "algorithm": alg,
                    "seed": seed,
                    "arm": arm,
                    "val_auc": 0.95,
                    "reliance": r,
                    "agreement_behaviour": ag,
                }
            )
    return pd.DataFrame(out)


SEP_OK = pd.DataFrame({"split": list("wxyz"), "meta_auc_original": 0.9, "meta_auc_augmented": 0.55})


def test_mechanism_supported() -> None:
    r = m.judge(rows(0.2, 0.7, 0.03), SEP_OK)
    assert r.f1 and r.f2 == "HOLDS" and r.f3 == "HOLDS" and r.f4 and r.supported


def test_mechanism_not_supported_when_linear_reliance_stays() -> None:
    r = m.judge(rows(0.8, 0.9, 0.0), SEP_OK.assign(meta_auc_augmented=0.7))
    assert not r.f1 and r.f2 == "DOES NOT HOLD" and not r.f4 and not r.supported


def test_not_testable_when_reliance_is_tiny() -> None:
    t = rows(0.2, 0.7, 0.03)
    t["reliance"] = t["reliance"] * 0.01
    assert m.judge(t, SEP_OK).f2 == "NOT TESTABLE"


def test_render_and_no_test_access() -> None:
    r = m.judge(rows(0.2, 0.7, 0.03), SEP_OK)
    assert "**Mechanism SUPPORTED**" in m.render(rows(0.2, 0.7, 0.03), SEP_OK, r, "h")
    src = inspect.getsource(m)
    assert "load_test" not in src and "locked_test" not in src
