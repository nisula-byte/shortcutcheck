"""Violation metrics with confidence intervals (roadmap step 2.6).

For one MR run:

* violation rate: share of counted inputs (changed, valid, realistic) that violate the MR;
* mean |delta p|: mean absolute change in malware probability over counted inputs.

Both get a percentile bootstrap 95% CI over counted inputs (seeded, 2,000 resamples by
default). The rate also gets a Wilson score interval: when there are no violations the
bootstrap interval collapses to [0, 0], which hides how few inputs were tested, while Wilson
still gives an honest upper bound (decision D14).

``summarise`` can be restricted to a subset of inputs with ``mask``: for example valid-year vs
invalid-year apps (decision D11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shortcutcheck.runner.engine import MRRun

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    @property
    def is_nan(self) -> bool:
        return math.isnan(self.low)


NAN = Interval(float("nan"), float("nan"))


@dataclass(frozen=True)
class MRSummary:
    mr_id: str
    n_counted: int
    n_violations: int
    n_unrealistic: int
    rate: float
    rate_ci: Interval
    rate_wilson: Interval
    delta: float
    delta_ci: Interval
    rate_unfiltered: float


def wilson(k: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return NAN
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> Interval:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return NAN
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 200):
        stop = min(start + 200, n_boot)
        idx = rng.integers(0, v.size, size=(stop - start, v.size))
        means[start:stop] = v[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return Interval(float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def summarise(
    run: MRRun,
    mask: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> MRSummary:
    """Violation rate and mean |delta p| with CIs, optionally over a subset of inputs."""
    keep = np.ones(run.n_inputs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    counted = run.counted & keep
    violated = (run.violated & counted)[counted].astype(float)
    delta = np.abs(run.p_after - run.p_before)[counted]
    n, k = int(counted.sum()), int(violated.sum())
    unfiltered = run.counted_unfiltered & keep
    n_unf = int(unfiltered.sum())
    return MRSummary(
        mr_id=run.mr_id,
        n_counted=n,
        n_violations=k,
        n_unrealistic=int((unfiltered & ~counted).sum()),
        rate=k / n if n else float("nan"),
        rate_ci=bootstrap_mean(violated, n_boot, seed),
        rate_wilson=wilson(k, n),
        delta=float(delta.mean()) if n else float("nan"),
        delta_ci=bootstrap_mean(delta, n_boot, seed + 1),
        rate_unfiltered=(int((run.violated & unfiltered).sum()) / n_unf if n_unf else float("nan")),
    )


def fmt_rate(s: MRSummary) -> str:
    """'0.123 [0.10, 0.15]' using the bootstrap CI; Wilson when there are no violations."""
    if math.isnan(s.rate):
        return "–"
    ci = s.rate_wilson if s.n_violations in (0, s.n_counted) else s.rate_ci
    return f"{s.rate:.3f} [{ci.low:.3f}, {ci.high:.3f}]"
