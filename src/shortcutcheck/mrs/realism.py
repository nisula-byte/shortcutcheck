"""Realism check for perturbed apps (roadmap step 2.4).

A perturbed app counts only if it still looks like a real app: its distance to the nearest
real training app must be no larger than what is normal between real apps.

Distance is Gower's distance for mixed data (Gower, 1971): the mean, over all features, of a
per-feature distance in [0, 1].

* 0/1 flags: 1 if they differ, else 0 (Hamming).
* Numeric features (``apk_size``, ``year``, SDK versions): absolute difference after mapping
  each value to its percentile among training apps, so heavy-tailed sizes cannot dominate.
* Missing values (one Dada app has "unknown" API flags) are treated as flag absent (0) and as
  the median for numeric features. This only affects distances, never model inputs.

Threshold: the ``percentile`` (default 95th) of nearest-neighbour distances between real
training apps, each measured to its nearest *other* training app, on a seeded sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from shortcutcheck.mrs.base import NotFittedError


@dataclass(frozen=True)
class _Space:
    binary: list[str]
    numeric: list[str]
    sorted_numeric: dict[str, np.ndarray]

    @property
    def n_features(self) -> int:
        return len(self.binary) + len(self.numeric)


class RealismCheck:
    def __init__(
        self,
        percentile: float = 95.0,
        reference_sample: int = 2000,
        seed: int = 0,
        chunk: int = 512,
    ) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100")
        self.percentile = percentile
        self.reference_sample = reference_sample
        self.seed = seed
        self.chunk = chunk
        self._space: _Space | None = None
        self._bin: np.ndarray = np.empty((0, 0))
        self._num: np.ndarray = np.empty((0, 0))
        self.reference_distances: np.ndarray = np.empty(0)
        self.threshold = float("nan")

    # ---- fitting ------------------------------------------------------------------------

    def fit(self, x_train: pd.DataFrame) -> Self:
        x = x_train.apply(pd.to_numeric, errors="coerce")
        binary = [c for c in x.columns if set(x[c].dropna().unique()) <= {0, 1}]
        numeric = [c for c in x.columns if c not in binary]
        self._space = _Space(
            binary=[str(c) for c in binary],
            numeric=[str(c) for c in numeric],
            sorted_numeric={str(c): np.sort(x[c].dropna().to_numpy(dtype=float)) for c in numeric},
        )
        self._bin, self._num = self._encode(x)
        n = len(x)
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(n, size=min(self.reference_sample, n), replace=False)
        self.reference_distances = self._nearest(
            self._bin[sample], self._num[sample], exclude=sample
        )
        self.threshold = float(np.percentile(self.reference_distances, self.percentile))
        return self

    # ---- public -------------------------------------------------------------------------

    def distances(self, x: pd.DataFrame) -> np.ndarray:
        """Gower distance from each row to its nearest real training app."""
        b, m = self._encode(x.apply(pd.to_numeric, errors="coerce"))
        return self._nearest(b, m)

    def is_realistic(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.distances(x) <= self.threshold, dtype=bool)

    # ---- internals ----------------------------------------------------------------------

    @property
    def space(self) -> _Space:
        if self._space is None:
            raise NotFittedError("RealismCheck has not been fitted")
        return self._space

    def _encode(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sp = self.space
        missing = [c for c in sp.binary + sp.numeric if c not in x.columns]
        if missing:
            raise ValueError(f"columns missing for the realism check: {missing[:5]}")
        b = x[sp.binary].fillna(0).to_numpy(dtype=np.float32)
        num = np.empty((len(x), len(sp.numeric)), dtype=np.float32)
        for j, col in enumerate(sp.numeric):
            ref = sp.sorted_numeric[col]
            v = x[col].to_numpy(dtype=float)
            # Mid-rank percentile in [0, 1]; missing -> median.
            lo = np.searchsorted(ref, v, side="left")
            hi = np.searchsorted(ref, v, side="right")
            q = (lo + hi) / (2.0 * max(len(ref), 1))
            num[:, j] = np.where(np.isnan(v), 0.5, q)
        return b, num

    def _nearest(
        self, b: np.ndarray, m: np.ndarray, exclude: np.ndarray | None = None
    ) -> np.ndarray:
        """Nearest-neighbour Gower distance to the training apps, in chunks."""
        ref_b, ref_m = self._bin, self._num
        ref_ones = ref_b.sum(axis=1)
        p = self.space.n_features
        out = np.empty(len(b), dtype=float)
        for start in range(0, len(b), self.chunk):
            qb = b[start : start + self.chunk]
            qm = m[start : start + self.chunk]
            # Hamming on 0/1 flags: |a| + |b| - 2 a.b (one matrix product).
            d = qb.sum(axis=1)[:, None] + ref_ones[None, :] - 2.0 * (qb @ ref_b.T)
            for j in range(qm.shape[1]):
                d += np.abs(qm[:, j][:, None] - ref_m[:, j][None, :])
            if exclude is not None:
                rows = np.arange(len(qb))
                d[rows, exclude[start : start + self.chunk]] = np.inf
            out[start : start + len(qb)] = d.min(axis=1) / p
        return out
