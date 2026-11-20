"""Roadmap step 2.6 Check: metrics on hand-computed toy examples."""

from __future__ import annotations

import numpy as np
import pytest

from shortcutcheck.runner import MRRun
from shortcutcheck.runner.stats import bootstrap_mean, fmt_rate, summarise, wilson


def test_wilson_matches_published_values() -> None:
    # Standard textbook values (Newcombe 1998): 0/10 and 5/10 at 95%.
    zero = wilson(0, 10)
    assert zero.low == 0.0 and zero.high == pytest.approx(0.2775, abs=1e-4)
    half = wilson(5, 10)
    assert half.low == pytest.approx(0.2366, abs=1e-4)
    assert half.high == pytest.approx(0.7634, abs=1e-4)
    assert wilson(0, 0).is_nan


def test_bootstrap_on_constant_and_known_data() -> None:
    assert bootstrap_mean(np.full(50, 0.3)).low == pytest.approx(0.3)
    assert bootstrap_mean(np.array([])).is_nan
    # 0/1 data with mean 0.5, n=400: CI close to the normal approximation 0.5 +/- 0.049.
    v = np.array([0, 1] * 200, dtype=float)
    ci = bootstrap_mean(v, n_boot=4000, seed=3)
    assert ci.low == pytest.approx(0.451, abs=0.01)
    assert ci.high == pytest.approx(0.549, abs=0.01)
    assert bootstrap_mean(v, seed=3) == bootstrap_mean(v, seed=3)  # seeded


def toy_run() -> MRRun:
    # Six inputs. Row 4 was not perturbed; row 5 is unrealistic.
    return MRRun(
        mr_id="MR1",
        p_before=np.array([0.9, 0.9, 0.2, 0.2, 0.9, 0.9]),
        p_after=np.array([0.1, 0.8, 0.7, 0.2, 0.1, 0.1]),
        perturbed=np.array([True, True, True, True, False, True]),
        valid=np.ones(6, bool),
        realistic=np.array([True, True, True, True, True, False]),
    )


def test_summarise_hand_computed() -> None:
    s = summarise(toy_run(), n_boot=500)
    # Counted rows 0-3; violations at rows 0 and 2.
    assert (s.n_counted, s.n_violations, s.n_unrealistic) == (4, 2, 1)
    assert s.rate == pytest.approx(0.5)
    assert s.delta == pytest.approx((0.8 + 0.1 + 0.5 + 0.0) / 4)
    assert s.rate_unfiltered == pytest.approx(3 / 5)  # row 5 flips too
    assert s.rate_ci.low <= s.rate <= s.rate_ci.high
    assert s.delta_ci.low <= s.delta <= s.delta_ci.high


def test_summarise_on_a_subset() -> None:
    # Only rows 1 and 3 (for example, invalid-year apps): neither flips.
    s = summarise(toy_run(), mask=np.array([False, True, False, True, False, False]))
    assert (s.n_counted, s.n_violations) == (2, 0)
    assert s.rate == 0.0 and s.rate_ci.high == 0.0
    # With no violations the report uses Wilson, which still shows an upper bound.
    assert fmt_rate(s).startswith("0.000 [0.000, 0.6")


def test_empty_summary() -> None:
    run = MRRun("MR1", np.zeros(2), np.zeros(2), np.zeros(2, bool), np.ones(2, bool))
    s = summarise(run)
    assert s.n_counted == 0 and np.isnan(s.rate) and s.rate_ci.is_nan
    assert fmt_rate(s) == "–"
