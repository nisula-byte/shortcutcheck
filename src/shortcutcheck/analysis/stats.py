"""Statistics for H2 and H3, fixed before any RQ result exists (PREREGISTRATION.md).

* ``holm``: Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1).
* ``cliffs_delta``: P(X > Y) - P(X < Y) over all pairs; +1 when every X beats every Y.
* ``wilcoxon_greater``: one-sided Wilcoxon signed-rank test that the differences are > 0,
  plus the two-sided p-value for reference. All-zero differences give p = 1.
* ``spearman``: Spearman rho; NaN when either variable is constant.
* ``cluster_bootstrap_rhos``: Spearman rho of several signals with one target, recomputed on
  resamples of whole clusters (configurations), the same resample for every signal so that
  differences between signals are paired.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def cliffs_delta(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float)[:, None]
    b = np.asarray(y, dtype=float)[None, :]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


@dataclass(frozen=True)
class Wilcoxon:
    n_pairs: int
    n_nonzero: int
    statistic: float
    p_greater: float
    p_two_sided: float


def wilcoxon_greater(differences: Sequence[float]) -> Wilcoxon:
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    nonzero = int((d != 0).sum())
    if nonzero == 0:
        return Wilcoxon(len(d), 0, float("nan"), 1.0, 1.0)
    kwargs = {"zero_method": "wilcox", "method": "auto"}
    greater = stats.wilcoxon(d, alternative="greater", **kwargs)
    both = stats.wilcoxon(d, alternative="two-sided", **kwargs)
    return Wilcoxon(
        len(d), nonzero, float(greater.statistic), float(greater.pvalue), float(both.pvalue)
    )


def spearman(x: Any, y: Any) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def cluster_bootstrap_rhos(
    table: pd.DataFrame,
    signals: Sequence[str],
    target: str,
    cluster: str,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """One row per resample, one column per signal: Spearman rho with ``target``.

    Each resample draws as many clusters as there are, with replacement, and keeps every row of
    each drawn cluster (a cluster drawn twice contributes its rows twice).
    """
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in table.groupby(cluster, sort=True)]
    values = {s: table[s].to_numpy(dtype=float) for s in signals}
    y = table[target].to_numpy(dtype=float)
    out = np.full((resamples, len(signals)), np.nan)
    positions = {idx: pos for pos, idx in enumerate(table.index)}
    pos_groups = [np.array([positions[i] for i in g]) for g in groups]
    for b in range(resamples):
        pick = rng.integers(0, len(pos_groups), size=len(pos_groups))
        rows = np.concatenate([pos_groups[i] for i in pick])
        for j, s in enumerate(signals):
            out[b, j] = spearman(values[s][rows], y[rows])
    return pd.DataFrame(out, columns=list(signals))


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(s, tail)), float(np.percentile(s, 100 - tail))


def bootstrap_p_greater(samples: np.ndarray) -> float:
    """One-sided bootstrap p-value for 'the statistic is > 0': (1 + #{<= 0}) / (1 + B)."""
    s = samples[~np.isnan(samples)]
    if s.size == 0:
        return float("nan")
    return float((1 + (s <= 0).sum()) / (1 + s.size))
