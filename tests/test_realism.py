"""Roadmap step 2.4: realism check (nearest real app, Gower distance, 95th percentile)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shortcutcheck.mrs import MetamorphicRelation, NotFittedError, RealismCheck, YearShift
from shortcutcheck.runner import MRRun, run_mr


def brute_gower(
    q: pd.DataFrame, ref: pd.DataFrame, binary: list[str], numeric: list[str]
) -> np.ndarray:
    """Slow reference implementation, written independently of the fast one."""
    pct = {}
    for c in numeric:
        r = np.sort(ref[c].to_numpy(dtype=float))
        pct[c] = lambda v, r=r: (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (
            2 * len(r)
        )
    out = []
    for _, a in q.iterrows():
        best = np.inf
        for _, b in ref.iterrows():
            d = sum(float(a[c] != b[c]) for c in binary)
            d += sum(abs(pct[c](a[c]) - pct[c](b[c])) for c in numeric)
            best = min(best, d / (len(binary) + len(numeric)))
        out.append(best)
    return np.array(out)


@st.composite
def frames(draw: st.DrawFn, n_min: int = 3) -> pd.DataFrame:
    n = draw(st.integers(n_min, 12))
    data = {f"f{i}": draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)) for i in range(5)}
    data["size"] = draw(st.lists(st.integers(2, 10**8), min_size=n, max_size=n))
    data["year"] = draw(st.lists(st.integers(2008, 2021), min_size=n, max_size=n))
    df = pd.DataFrame(data)
    df.loc[0, "f0"], df.loc[1, "f0"] = 0, 1  # keep f* recognisably binary
    return df


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(train=frames(), query=frames(n_min=2))
def test_fast_distance_matches_brute_force(train: pd.DataFrame, query: pd.DataFrame) -> None:
    check = RealismCheck(chunk=3).fit(train)
    binary = [f"f{i}" for i in range(5) if set(train[f"f{i}"]) <= {0, 1}]
    numeric = [c for c in train.columns if c not in binary]
    assert check.space.binary == binary and check.space.numeric == numeric
    expected = brute_gower(query, train, binary, numeric)
    np.testing.assert_allclose(check.distances(query), expected, atol=1e-6)


def test_hand_computed_distance() -> None:
    train = pd.DataFrame({"a": [0, 1, 1, 0], "b": [0, 0, 1, 1], "size": [10, 20, 30, 40]})
    check = RealismCheck(reference_sample=4).fit(train)
    # A copy of a training app is at distance 0.
    assert check.distances(train.iloc[[2]])[0] == 0.0
    # (a=0, b=0, size=30): nearest is row 0 (flags equal, size percentile 0.625 vs 0.125)
    # or row 2 (a, b differ: 2, size equal): min((0 + 0.5) / 3, 2 / 3) = 1/6.
    q = pd.DataFrame({"a": [0], "b": [0], "size": [30]})
    assert check.distances(q)[0] == pytest.approx(1 / 6)


def test_threshold_uses_distance_to_another_training_app() -> None:
    train = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1]})
    check = RealismCheck(percentile=50, reference_sample=4).fit(train)
    # Every app's nearest other app differs in one of two flags: distance 0.5, never 0.
    assert set(check.reference_distances) == {0.5}
    assert check.threshold == 0.5
    far = pd.DataFrame({"a": [1], "b": [1]})
    assert check.is_realistic(far).all()  # it is itself a training app


def test_far_away_perturbation_is_unrealistic() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(400, 20))
    base[:, :10] = 0  # the first ten flags are always off in real apps
    train = pd.DataFrame(base, columns=[f"f{i}" for i in range(20)])
    check = RealismCheck(seed=1).fit(train)
    weird = train.iloc[:5].copy()
    weird.iloc[:, :10] = 1
    assert not check.is_realistic(weird).any()
    assert check.is_realistic(train.iloc[:50]).mean() > 0.9


def test_missing_values_do_not_break_distances() -> None:
    train = pd.DataFrame({"a": [0, 1, np.nan], "size": [1.0, 2.0, np.nan]})
    d = RealismCheck(reference_sample=3).fit(train).distances(train)
    assert np.isfinite(d).all()


def test_errors() -> None:
    with pytest.raises(NotFittedError):
        RealismCheck().distances(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="percentile"):
        RealismCheck(percentile=100)
    check = RealismCheck(reference_sample=2).fit(pd.DataFrame({"a": [0, 1], "b": [1, 0]}))
    with pytest.raises(ValueError, match="columns missing"):
        check.distances(pd.DataFrame({"a": [1]}))


class FlagFlip(MetamorphicRelation):
    """Turns on ten flags that real apps never have: always unrealistic."""

    mr_id = "FLIP"
    name = "test double"
    justification = "test double"

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        x[list(self.columns_touched)] = 1
        return x


class Sum:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = (x.sum(axis=1) > 10).astype(float).to_numpy()
        return np.column_stack([1 - p, p])


def test_runner_excludes_unrealistic_rows_and_keeps_the_unfiltered_rate() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(300, 20))
    base[:, :10] = 0
    x = pd.DataFrame(base, columns=[f"f{i}" for i in range(20)])
    cols = tuple(f"f{i}" for i in range(10))
    mr = FlagFlip(cols).fit(pd.concat([x, x.assign(**{c: 1 for c in cols}).iloc[:1]]))
    realism = RealismCheck(seed=0).fit(x)
    run = run_mr(Sum(), mr, x, seed=0, realism=realism)
    assert run.n_unrealistic == 300 and run.n_counted == 0
    assert np.isnan(run.violation_rate)
    assert run.violation_rate_unfiltered > 0.5
    plain = run_mr(Sum(), mr, x, seed=0)
    assert plain.realistic is None and plain.n_unrealistic == 0


def test_realistic_mr1_keeps_most_rows() -> None:
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"year": rng.integers(2010, 2018, 500), "f": rng.integers(0, 2, 500), "g": 0})
    mr = YearShift().fit(x)
    run = run_mr(Sum(), mr, x, seed=0, realism=RealismCheck(seed=0).fit(x))
    assert run.n_unrealistic / run.perturbed.sum() < 0.1


def test_unrealistic_count_on_a_hand_built_run() -> None:
    run = MRRun(
        mr_id="MR1",
        p_before=np.array([0.9, 0.9, 0.1]),
        p_after=np.array([0.1, 0.1, 0.1]),
        perturbed=np.ones(3, bool),
        valid=np.ones(3, bool),
        realistic=np.array([True, False, True]),
    )
    assert run.n_unrealistic == 1 and run.n_counted == 2
    assert run.violation_rate == pytest.approx(1 / 2)
    assert run.violation_rate_unfiltered == pytest.approx(2 / 3)
