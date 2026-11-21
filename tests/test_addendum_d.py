"""Addendum D: decision rules on hand-built tables with known answers, and the dry-run guard."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from shortcutcheck.addenda import addendum_d as d
from shortcutcheck.evaluation import addendum_d_run as run

ALGOS = ["lr", "svm", "rf", "lgbm", "mlp"]
SEEDS = [11, 22, 33, 44, 55]


def h4_rows(effect: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    """effect[split] = (MR4 ratio, F1 drop, MR1 drop, contrary-accuracy gain)."""
    rng = np.random.default_rng(1)
    out = []
    for split, (alg, seed) in itertools.product(effect, itertools.product(ALGOS, SEEDS)):
        ratio, f1_drop, held, gain = effect[split]
        mr4 = 0.2 + rng.uniform(0, 0.02)
        n1, n2 = rng.uniform(-0.002, 0.002, 2)
        arms = {
            "baseline": {"rate_MR4": mr4, "f1": 0.9, "rate_MR1": 0.1, "contrary_acc": 0.8},
            "augmented": {
                "rate_MR4": mr4 * ratio,
                "f1": 0.9 - f1_drop,
                "rate_MR1": 0.1 - held + n1,
                "contrary_acc": 0.8 + gain + (n2 if gain else 0.0),
            },
            "duplicate": {"rate_MR4": mr4, "f1": 0.9, "rate_MR1": 0.1, "contrary_acc": 0.8},
        }
        for arm, m in arms.items():
            out.append(
                {"split": split, "algorithm": alg, "seed": seed, "arm": arm, "rate_MR3": 0.01, **m}
            )
    return pd.DataFrame(out)


GOOD = (0.2, 0.01, 0.05, 0.05)


@pytest.mark.parametrize(
    ("effect", "verdict", "passing", "n_d"),
    [
        ({s: GOOD for s in "wxyz"}, "SUPPORTED", 4, 4),
        (
            {"w": GOOD, "x": GOOD, "y": (0.2, 0.05, 0.05, 0.0), "z": (0.7, 0.0, 0.05, 0.0)},
            "PARTIALLY SUPPORTED",
            2,
            2,
        ),
        ({s: (0.2, 0.01, 0.0, 0.0) for s in "wxyz"}, "NOT SUPPORTED", 0, 0),
    ],
)
def test_h4_rules(
    effect: dict[str, tuple[float, float, float, float]], verdict: str, passing: int, n_d: int
) -> None:
    r = d.judge_h4(h4_rows(effect))
    assert r.verdict == verdict
    assert r.n_passing == passing
    assert r.n_h4d == n_d


def test_h4a_needs_both_significance_and_halving() -> None:
    r = d.judge_h4(h4_rows({s: (0.6, 0.0, 0.05, 0.0) for s in "wxyz"}))
    assert all(s.p_mr4_holm < 0.05 for s in r.splits)  # significant ...
    assert not any(s.a for s in r.splits)  # ... but not halved


def h5a_rows(gap: float) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    out = []
    for alg, seed in itertools.product(ALGOS, SEEDS):
        base = 0.3 + rng.uniform(0, 0.05)
        for arm, decay in (
            ("baseline", base),
            ("augmented", base - gap + rng.uniform(-0.005, 0.005)),
            ("duplicate", base),
        ):
            out.append(
                {
                    "split": "s",
                    "algorithm": alg,
                    "seed": seed,
                    "arm": arm,
                    "decay_1516": decay,
                    "decay_1719": decay,
                    "f1_val": 0.9,
                    "f1_holdout": 0.9 - decay,
                    "f1_1719": 0.9 - decay,
                    "rate_MR4_holdout": 0.05,
                }
            )
    return pd.DataFrame(out)


def test_h5a_holds_only_when_augmented_decays_less() -> None:
    assert d.judge_h5a(h5a_rows(0.05)).holds
    assert not d.judge_h5a(h5a_rows(0.0)).holds
    assert not d.judge_h5a(h5a_rows(-0.05)).holds


def h5b_table(strength: float) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    out = []
    for alg, skew, setting, seed in itertools.product(ALGOS, (0.0, 0.5), range(6), SEEDS):
        decay = rng.uniform(0.1, 0.7)
        out.append(
            {
                "model": f"{alg}-{skew}-{setting}-{seed}",
                "algorithm": alg,
                "skew": skew,
                "setting": setting,
                "seed": seed,
                "decay": decay,
                "mr4_monitor": strength * decay + rng.normal(0, 0.05),
                "psi": rng.uniform(0, 1),
                "confidence_drop": rng.uniform(0, 1),
                "atc_drop": rng.uniform(0, 1),
                "mr4_pre": rng.uniform(0, 1),
            }
        )
    return pd.DataFrame(out)


def test_h5b_and_h5c_rules() -> None:
    strong = d.judge_h5b(h5b_table(1.0), resamples=300)
    assert strong.h5b
    assert all(strong.h5c().values())
    none = d.judge_h5b(h5b_table(0.0), resamples=300)
    assert not none.h5b
    assert not any(none.h5c().values())
    assert set(strong.pre) == {"un-injected", "full pool"}


def test_reports_state_verdicts() -> None:
    rows = h4_rows({s: GOOD for s in "wxyz"})
    for c in ("roc_auc", "rate_MR5", "rate_MR6"):
        rows[c] = 0.5
    assert "**H4: SUPPORTED**" in d.render_h4(rows, d.judge_h4(rows), "head")
    text = d.render_h5(
        h5a_rows(0.05),
        d.judge_h5a(h5a_rows(0.05)),
        d.judge_h5b(h5b_table(1.0), resamples=200),
        "head",
    )
    assert "H5a (the fix reduces decay): HOLDS" in text
    assert "H5b (MR4 at monitoring time predicts decay): HOLDS" in text


def test_dry_run_never_opens_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("load_test called in a dry run")

    monkeypatch.setattr(run, "load_test", forbidden)
    monkeypatch.setattr(run, "load_part", lambda _cfg, split, part: f"{split}:{part}")
    assert run.open_frame(None, "rq12-DR-AG", "x", dry_run=True) == "rq12-DR-AG:val"  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        run.open_frame(None, "rq12-DR-AG", "x", dry_run=False)  # type: ignore[arg-type]


def test_duplicate_arm_doubles_unchanged() -> None:
    x = pd.DataFrame({"year": [2012, 2013], "f": [0, 1]})
    y = pd.Series([0, 1])
    tr = run.TrainData(x, y, {}, None)  # type: ignore[arg-type]
    xd, yd, rej = run.arm_data(tr, "duplicate", 11)
    assert len(xd) == 4 and rej == 0
    pd.testing.assert_frame_equal(xd.iloc[2:].reset_index(drop=True), x)
    assert run.arm_data(tr, "baseline", 11)[0] is x
    with pytest.raises(ValueError):
        run.arm_data(tr, "other", 11)
