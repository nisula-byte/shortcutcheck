"""Roadmap step 2.5: oracle models and their exact expected answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shortcutcheck.mrs import MetadataSwap, SizeScale, YearShift
from shortcutcheck.runner import run_mr
from shortcutcheck.runner.known_answers import judge
from shortcutcheck.runner.oracles import (
    Constant,
    Stump,
    expected_mr1,
    expected_mr3,
    expected_mr4,
)

VALID = (2008, 2021)


def apps(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n))
    x = pd.DataFrame(
        {
            "year": np.where(y == 1, rng.integers(2010, 2015, n), rng.integers(2013, 2019, n)),
            "apk_size": np.where(
                y == 1, rng.integers(10_000, 3_000_000, n), rng.integers(1_000_000, 50_000_000, n)
            ),
            "send_sms": rng.integers(0, 2, n),
        }
    )
    x.loc[:50, "year"] = 1980  # some invalid years, as in Dada
    return x, y


def test_stump_learns_one_split() -> None:
    x, y = apps()
    stump = Stump("year").fit(x, y)
    assert 2008 < stump.threshold < 2019 and stump.left_class != stump.right_class
    with pytest.raises(ValueError, match="no useful split"):
        Stump("year").fit(x.assign(year=2015), y)


@pytest.mark.parametrize("seed", [1, 2])
def test_expected_mr1_matches_simulation(seed: int) -> None:
    x, y = apps()
    stump = Stump("year").fit(x, y)
    mr = YearShift("year", VALID).fit(x)
    exp = expected_mr1(stump, mr, x["year"].to_numpy(dtype=float))
    observed = run_mr(stump, mr, x, seed=seed).violation_rate
    assert observed == pytest.approx(exp.mean(), abs=4 * np.sqrt(np.sum(exp * (1 - exp))) / len(x))


def test_expected_mr3_matches_simulation() -> None:
    x, y = apps()
    stump = Stump("apk_size").fit(x, y)
    mr = SizeScale().fit(x)
    exp = expected_mr3(stump, mr, x["apk_size"].to_numpy(dtype=float))
    assert 0.05 < exp.mean() < 0.95
    run = run_mr(stump, mr, x, seed=4)
    p = exp[run.counted]
    tol = 4 * np.sqrt(np.sum(p * (1 - p))) / len(p)
    assert run.violation_rate == pytest.approx(p.mean(), abs=tol)


def test_expected_mr4_matches_simulation() -> None:
    x, y = apps()
    stump = Stump("year").fit(x, y)
    mr = MetadataSwap(["year", "apk_size"], "year", VALID).fit(x, y)
    exp = expected_mr4(stump, mr, x["year"].to_numpy(dtype=float))
    run = run_mr(stump, mr, x, seed=5)
    p = exp[run.counted]
    assert run.violation_rate == pytest.approx(
        p.mean(), abs=4 * np.sqrt(np.sum(p * (1 - p))) / len(p)
    )


def test_judge_passes_on_correct_oracles() -> None:
    x, y = apps()
    mr1 = YearShift("year", VALID).fit(x)
    mr3 = SizeScale().fit(x)
    year, size, const = Stump("year").fit(x, y), Stump("apk_size").fit(x, y), Constant().fit(x, y)
    answers = [
        judge("toy", "year stump", year, mr1, x, 11),  # expected
        judge("toy", "year stump", year, mr3, x, 11),  # exact 0
        judge("toy", "size stump", size, mr1, x, 11),  # exact 0
        judge("toy", "size stump", size, mr3, x, 11),  # expected
        judge("toy", "constant", const, mr1, x, 11),  # exact 0
    ]
    assert [a.kind for a in answers] == ["expected", "exact 0", "exact 0", "expected", "exact 0"]
    assert all(a.passes for a in answers), answers
    assert answers[0].observed > 0.1


class Cheater(Stump):
    """Claims to read only `year` but actually uses `apk_size` too: the check must catch it."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        base = super().predict_proba(x)[:, 1]
        p = np.where(x.get("apk_size", pd.Series(0, index=x.index)) > 5_000_000, 1 - base, base)
        return np.column_stack([1 - p, p])


def test_a_model_that_reads_an_undeclared_column_fails() -> None:
    x, y = apps()
    cheat = Cheater("year").fit(x, y)
    cheat.feature_names_in_ = np.array(["year", "apk_size"])
    answer = judge("toy", "year stump", cheat, SizeScale().fit(x), x, 11)
    assert answer.kind == "exact 0" and not answer.passes


def test_an_mr_that_breaks_its_sampling_rule_fails() -> None:
    """MR1 promises a uniform draw over other valid years; a biased draw must be caught."""
    x, y = apps()
    stump = Stump("year").fit(x, y)
    assert judge("toy", "year stump", stump, YearShift("year", VALID).fit(x), x, 11).passes

    class Biased(YearShift):
        def _transform(
            self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
        ) -> pd.DataFrame:
            x[self.column] = self.valid_years.max()  # always the latest year
            return x

    answer = judge("toy", "year stump", stump, Biased("year", VALID).fit(x), x, 11)
    assert answer.kind == "expected" and not answer.passes
