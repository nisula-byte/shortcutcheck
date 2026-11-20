"""Realism check for perturbed apps (roadmap step 2.4).

A perturbed app counts only if it still looks like a real app: its distance to the nearest
real training app must be no larger than what is normal between real apps.

Distance is a weighted Gower distance for mixed data (Gower, 1971): a weighted mean, over
features, of a per-feature distance in [0, 1].

* 0/1 flags: 1 if they differ, else 0 (Hamming).
* Numeric features (``apk_size``, ``year``, SDK versions): absolute difference after mapping
  each value to its percentile among training apps, so heavy-tailed sizes cannot dominate.
* Missing values (one Dada app has "unknown" API flags) are treated as flag absent (0) and as
  the median for numeric features. This only affects distances, never model inputs.

Weights (decision D13): given ``metadata_columns``, the metadata block and the behaviour block
each carry ``metadata_weight`` and ``1 - metadata_weight`` of the distance (default 0.5 each),
split equally among the columns of each block. With plain Gower weights (each column 1/p), a
change to one metadata column moves the distance by at most 1/206 on Dada, so the filter
ignored metadata MRs: 91-100% of the perturbed apps it excluded were already beyond the
threshold before they were perturbed. With equal block weights, a perturbed app counts as
realistic only if some real app has both similar behaviour and similar metadata. Columns that
are constant in training carry no information and get weight 0.

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
    w_binary: np.ndarray
    w_numeric: np.ndarray


class RealismCheck:
    def __init__(
        self,
        percentile: float = 95.0,
        reference_sample: int = 2000,
        seed: int = 0,
        chunk: int = 512,
        metadata_columns: list[str] | None = None,
        metadata_weight: float = 0.5,
    ) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100")
        if not 0 < metadata_weight < 1:
            raise ValueError("metadata_weight must be between 0 and 1")
        self.metadata_columns = metadata_columns
        self.metadata_weight = metadata_weight
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
        varying = [str(c) for c in x.columns if x[c].nunique(dropna=True) > 1]
        binary = [c for c in varying if set(x[c].dropna().unique()) <= {0, 1}]
        numeric = [c for c in varying if c not in binary]
        weights = self._weights(varying)
        self._space = _Space(
            binary=binary,
            numeric=numeric,
            sorted_numeric={c: np.sort(x[c].dropna().to_numpy(dtype=float)) for c in numeric},
            w_binary=np.array([weights[c] for c in binary], dtype=np.float32),
            w_numeric=np.array([weights[c] for c in numeric], dtype=np.float32),
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

    def _weights(self, columns: list[str]) -> dict[str, float]:
        """Per-column weights summing to 1 (plain Gower, or equal metadata/behaviour blocks)."""
        if not columns:
            raise ValueError("no varying columns to measure distance on")
        meta = [c for c in columns if self.metadata_columns and c in self.metadata_columns]
        rest = [c for c in columns if c not in meta]
        if not meta or not rest:
            return {c: 1.0 / len(columns) for c in columns}
        w = {c: self.metadata_weight / len(meta) for c in meta}
        w.update({c: (1.0 - self.metadata_weight) / len(rest) for c in rest})
        return w

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
        """Nearest-neighbour weighted Gower distance to the training apps, in chunks."""
        sp = self.space
        ref_b, ref_m = self._bin, self._num
        ref_w = ref_b @ sp.w_binary
        out = np.empty(len(b), dtype=float)
        for start in range(0, len(b), self.chunk):
            qb = b[start : start + self.chunk]
            qm = m[start : start + self.chunk]
            # Weighted Hamming on 0/1 flags: w.a + w.b - 2 (w*a).b (one matrix product).
            d = (qb @ sp.w_binary)[:, None] + ref_w[None, :] - 2.0 * ((qb * sp.w_binary) @ ref_b.T)
            for j in range(qm.shape[1]):
                d += sp.w_numeric[j] * np.abs(qm[:, j][:, None] - ref_m[:, j][None, :])
            if exclude is not None:
                rows = np.arange(len(qb))
                d[rows, exclude[start : start + self.chunk]] = np.inf
            out[start : start + len(qb)] = np.maximum(d.min(axis=1), 0.0)
        return out
