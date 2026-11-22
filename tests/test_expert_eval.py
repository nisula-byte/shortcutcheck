"""Step 4.6: the questionnaire analysis on responses with known answers."""

from __future__ import annotations

import pandas as pd
import pytest

from shortcutcheck import expert_eval


def responses(useful: list[int], n: int | None = None) -> pd.DataFrame:
    n = n or len(useful)
    rows = []
    for i in range(n):
        u = useful[i]
        rows.append(
            {
                "participant": f"P{i + 1:02d}",
                "U1": u,
                "U2": u,
                "U3": u,
                "C1": 4,
                "C2": 5,
                "C3": 3,
                "I1": 3,
                "I2": 4,
                "I3": 5,
                "task1": "correct",
                "task2": "correct" if i % 2 else "partly",
            }
        )
    return pd.DataFrame(rows)


def test_nfr8_met_and_medians() -> None:
    r = expert_eval.analyse(responses([4, 5, 4, 3, 5, 4]))
    assert r.usefulness_median == 4.0 and r.nfr8
    assert {c.name: c.median for c in r.constructs}["understandability"] == 4.0
    assert r.task_correct == {"task1": 1.0, "task2": 0.5}
    assert r.enough_participants
    assert "NFR8 (median usefulness >= 4): met" in expert_eval.render(r)


def test_nfr8_not_met_and_too_few() -> None:
    r = expert_eval.analyse(responses([3, 3, 4]))
    assert not r.nfr8 and not r.enough_participants
    assert "fewer than the planned 5" in expert_eval.render(r)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.assign(U1=6), "1 to 5"),
        (lambda f: f.drop(columns=["I3"]), "missing"),
        (lambda f: f.assign(participant="P01"), "twice"),
        (lambda f: f.assign(task1="yes"), "correct, partly or incorrect"),
    ],
)
def test_bad_responses_are_rejected(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        expert_eval.analyse(change(responses([4, 4, 4, 4, 4])))
