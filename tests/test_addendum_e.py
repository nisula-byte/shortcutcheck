"""Addendum E: the adjusted correlation, the E5 rule and the verdict, on known answers."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from shortcutcheck.addenda import addendum_e as e
from shortcutcheck.evaluation import addendum_e_run as run

ALGOS = ["lr", "svm", "rf", "lgbm", "mlp"]


def test_adjusted_rho_removes_between_group_differences() -> None:
    # Within each group the signal rises with the target; between groups it falls.
    groups = np.repeat(np.arange(4), 10)
    target = np.concatenate([np.arange(10) + 100 * g for g in range(4)]).astype(float)
    signal = np.concatenate([np.arange(10) - 100 * g for g in range(4)]).astype(float)
    raw = pd.Series(signal).corr(pd.Series(target), method="spearman")
    assert raw < -0.5
    assert e.adjusted_rho(signal, target, groups) == pytest.approx(1.0)


def test_adjusted_rho_is_zero_without_within_group_signal() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 40)
    target = rng.normal(size=200) + groups * 10
    signal = rng.normal(size=200) + groups * 10  # only the group means agree
    assert abs(e.adjusted_rho(signal, target, groups)) < 0.2
    assert pd.Series(signal).corr(pd.Series(target), method="spearman") > 0.8


def pool(strength: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for alg, setting, s in itertools.product(ALGOS, range(6), [11, 22, 33, 44, 55]):
        decay = rng.uniform(0.0, 0.2) + ALGOS.index(alg) * 0.1
        out.append(
            {
                "model": f"{alg}-h{setting}-s{s}",
                "algorithm": alg,
                "setting": setting,
                "seed": s,
                "decay": decay,
                "mr4_monitor": strength * decay
                + rng.normal(0, 0.01)
                + (1 - strength) * rng.uniform(0, 1),
                "psi": rng.uniform(0, 1),
                "confidence_drop": rng.uniform(0, 1),
                "atc_drop": rng.uniform(0, 1),
            }
        )
    return pd.DataFrame(out)


def test_judge_adjusted_detects_within_algorithm_signal() -> None:
    strong = e.judge_adjusted(e.pool_table(pool(1.0)), "x", resamples=200)
    assert strong.holds and strong.rho > 0.8
    none = e.judge_adjusted(e.pool_table(pool(0.0)), "x", resamples=200)
    assert not none.holds


def arms(gain_f1: float) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    out = []
    for alg, s in itertools.product(ALGOS, [11, 22, 33, 44, 55]):
        base = 0.85 + rng.uniform(0, 0.01)
        for arm, f1h in (
            ("baseline", base - 0.08),
            ("augmented", base - 0.05),
            ("duplicate", base - 0.08),
            ("behaviour", base - 0.05 - gain_f1 + rng.uniform(-0.002, 0.002)),
        ):
            out.append(
                {
                    "split": "s",
                    "algorithm": alg,
                    "seed": s,
                    "arm": arm,
                    "f1_val": base,
                    "f1_holdout": f1h,
                    "decay_1516": base - f1h,
                    "f1_1719": f1h - 0.3,
                    "decay_1719": base - f1h + 0.3,
                    "rate_MR4_holdout": 0.0 if arm == "behaviour" else 0.05,
                }
            )
    return pd.DataFrame(out)


@pytest.mark.parametrize(
    ("gain", "verdict"),
    [(0.03, "fix better"), (-0.03, "behaviour-only better"), (0.0, "no difference")],
)
def test_e5_verdicts(gain: float, verdict: str) -> None:
    rows = arms(gain)
    if gain == 0.0:  # exactly equal pairs: no difference
        b = rows["arm"] == "behaviour"
        rows.loc[b, "f1_holdout"] = rows.loc[rows["arm"] == "augmented", "f1_holdout"].to_numpy()
        rows.loc[b, "decay_1516"] = rows.loc[rows["arm"] == "augmented", "decay_1516"].to_numpy()
    t = e.judge_e5(rows)
    assert set(t["verdict"]) == {verdict}


def test_overall_verdict_and_report() -> None:
    r = e.judge(arms(0.03), pool(1.0), resamples=200)
    assert r.e1.holds and r.e2.h5b
    assert r.verdict == "REPLICATES"
    text = e.render(arms(0.03), pool(1.0), r, "head")
    assert "**Replication (E1 and E2): REPLICATES.**" in text
    weak = e.judge(arms(0.03), pool(0.0), resamples=200)
    assert weak.verdict == "PARTIALLY REPLICATES"


def test_behaviour_arm_drops_metadata() -> None:
    x = pd.DataFrame({"year": [2012, 2013], "apk_size": [1, 2], "perm": [0, 1]})
    tr = run.TrainData(x, pd.Series([0, 1]), {}, None)  # type: ignore[arg-type]
    xb, _, rej = run.arm(tr, "behaviour", 11, ["year", "apk_size"])
    assert list(xb.columns) == ["perm"] and rej == 0


def test_dry_run_never_opens_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("load_test called in a dry run")

    monkeypatch.setattr(run, "load_test", forbidden)
    monkeypatch.setattr(run, "load_part", lambda _c, split, part: f"{split}:{part}")
    assert run.open_test(None, "h", "s", "x", dry_run=True) == "s:val"  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        run.open_test(None, "h", "s", "x", dry_run=False)  # type: ignore[arg-type]


def test_rq3_settings_match_the_pool(tmp_path: object) -> None:
    from pathlib import Path

    hyper, bins, floor = run.rq3_settings(Path(__file__).resolve().parents[1])
    assert set(hyper) == set(ALGOS) and all(len(v) == 6 for v in hyper.values())
    assert bins == 10 and floor == pytest.approx(1e-4)
