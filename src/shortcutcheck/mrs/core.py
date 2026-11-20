"""Core metamorphic relations MR1-MR4 (roadmap step 2.2).

Every perturbed value comes from what the training inputs contain: MR1 and MR2 draw observed
values, MR3 clips to the observed size range, and MR4 copies a real training app's metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError, changed_rows


def _draw_other(current: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each row, draw a pool index uniformly, never the row's own value if it is in the pool.

    ``current`` holds each row's index into ``pool``, or -1 if its value is not in the pool.
    """
    n = len(pool)
    own = current >= 0
    draw = np.empty(len(current), dtype=int)
    draw[~own] = rng.integers(0, n, size=int((~own).sum()))
    if own.any():
        shifted = rng.integers(0, n - 1, size=int(own.sum()))
        shifted[shifted >= current[own]] += 1
        draw[own] = shifted
    return draw


class YearShift(MetamorphicRelation):
    """MR1: move the build year to another valid year seen in training."""

    mr_id = "MR1"
    name = "Year shift"
    justification = (
        "The year an app was built is not behaviour: the same code is equally malicious or "
        "benign whatever year its build tool stamped on it."
    )

    def __init__(self, column: str = "year", valid: tuple[int, int] = (2008, 2021)) -> None:
        super().__init__((column,))
        self.column = column
        self.valid = valid
        self.valid_years: np.ndarray = np.empty(0)

    def _fit(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:
        values = np.asarray(self.ranges[self.column].values)
        keep = (values >= self.valid[0]) & (values <= self.valid[1])
        self.valid_years = values[keep]
        if len(self.valid_years) < 2:
            raise NotApplicableError(
                f"MR1 needs at least two valid years in training, found {list(self.valid_years)}"
            )

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        year = pd.to_numeric(x[self.column], errors="coerce").to_numpy(dtype=float)
        current = np.searchsorted(self.valid_years, year)
        current = np.where(
            (current < len(self.valid_years))
            & (self.valid_years[np.minimum(current, len(self.valid_years) - 1)] == year),
            current,
            -1,
        )
        x[self.column] = self.valid_years[_draw_other(current, self.valid_years, rng)]
        return x

    def is_valid(self, x_new: pd.DataFrame) -> np.ndarray:
        """Valid only if the new year is one of the valid training years (never 1980 or 2107)."""
        year = pd.to_numeric(x_new[self.column], errors="coerce").to_numpy(dtype=float)
        return np.isin(year, self.valid_years)


class SdkShift(MetamorphicRelation):
    """MR2: replace (minSdkVersion, targetSdkVersion) with another observed pair."""

    mr_id = "MR2"
    name = "SDK version shift"
    justification = (
        "The SDK versions an app declares reflect the toolchain era it was built in, not "
        "what its code does."
    )

    def __init__(
        self,
        min_column: str = "minSdkVersion",
        target_column: str = "targetSdkVersion",
        min_known_share: float = 0.05,
    ) -> None:
        super().__init__((min_column, target_column))
        self.min_known_share = min_known_share
        self.pairs: np.ndarray = np.empty((0, 2))
        self.known_share = 0.0

    def _fit(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:
        pairs = x_train[list(self.columns_touched)].apply(pd.to_numeric, errors="coerce")
        known = (pairs > 0).all(axis=1) & (pairs.iloc[:, 0] <= pairs.iloc[:, 1])
        self.known_share = float(known.mean())
        if self.known_share < self.min_known_share:
            raise NotApplicableError(
                f"MR2: only {self.known_share:.2%} of training apps have known SDK versions "
                f"(0 means missing); at least {self.min_known_share:.0%} needed"
            )
        self.pairs = np.unique(pairs[known].to_numpy(dtype=float), axis=0)
        if len(self.pairs) < 2:
            raise NotApplicableError("MR2 needs at least two distinct observed SDK pairs")

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        lookup = {tuple(p): i for i, p in enumerate(self.pairs.tolist())}
        rows = x[list(self.columns_touched)].apply(pd.to_numeric, errors="coerce")
        current = np.array([lookup.get(tuple(r), -1) for r in rows.to_numpy(dtype=float).tolist()])
        chosen = self.pairs[_draw_other(current, self.pairs, rng)]
        for j, col in enumerate(self.columns_touched):
            x[col] = chosen[:, j]
        return x

    def is_valid(self, x_new: pd.DataFrame) -> np.ndarray:
        rows = x_new[list(self.columns_touched)].apply(pd.to_numeric, errors="coerce")
        known = {tuple(p) for p in self.pairs.tolist()}
        return np.array([tuple(r) in known for r in rows.to_numpy(dtype=float).tolist()])


class SizeScale(MetamorphicRelation):
    """MR3: rescale the APK size by a factor between 0.5 and 2, clipped to the training range."""

    mr_id = "MR3"
    name = "APK size rescale"
    justification = (
        "APK size mostly reflects bundled assets such as images and libraries, not whether "
        "the code is malicious."
    )

    def __init__(self, column: str = "apk_size", factor: tuple[float, float] = (0.5, 2.0)) -> None:
        if not 0 < factor[0] < 1 < factor[1]:
            raise ValueError("factor range must straddle 1, e.g. (0.5, 2.0)")
        super().__init__((column,))
        self.column = column
        self.factor = factor

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        observed = self.ranges[self.column]
        # Log-uniform, so halving and doubling are equally likely.
        log_f = rng.uniform(np.log(self.factor[0]), np.log(self.factor[1]), size=len(x))
        size = pd.to_numeric(x[self.column], errors="coerce").to_numpy(dtype=float)
        x[self.column] = np.clip(np.round(size * np.exp(log_f)), observed.low, observed.high)
        return x


class MetadataSwap(MetamorphicRelation):
    """MR4: give each app the full metadata of a real training app from the other class.

    "Other class" is relative to the model's own prediction for the app, so the MR needs no
    labels for the inputs it tests. Donors are training apps (training labels are known) whose
    year is valid.
    """

    mr_id = "MR4"
    name = "Full metadata swap"
    justification = (
        "Counterfactual: the same behaviour wearing the other class's metadata. If the "
        "decision follows the metadata, the model relies on it rather than on behaviour."
    )

    def __init__(
        self,
        columns: Sequence[str],
        year_column: str | None = "year",
        valid: tuple[int, int] = (2008, 2021),
    ) -> None:
        super().__init__(tuple(columns))
        self.year_column = year_column if year_column in columns else None
        self.valid = valid
        self.donors: dict[int, np.ndarray] = {}

    def _fit(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:
        if y_train is None:
            raise NotApplicableError("MR4 needs training labels to build its donor pools")
        meta = x_train[list(self.columns_touched)].apply(pd.to_numeric, errors="coerce")
        ok = np.array(meta.notna().all(axis=1).to_numpy(), dtype=bool)
        if self.year_column is not None:
            year = meta[self.year_column].to_numpy(dtype=float)
            ok &= (year >= self.valid[0]) & (year <= self.valid[1])
        labels = np.asarray(y_train, dtype=int)
        for cls in (0, 1):
            pool = meta.to_numpy(dtype=float)[ok & (labels == cls)]
            if len(pool) == 0:
                raise NotApplicableError(f"MR4: no valid donor apps of class {cls}")
            self.donors[cls] = pool

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        if predicted is None or len(predicted) != len(x):
            raise ValueError("MR4 needs the model's predicted class for every row")
        target = 1 - np.asarray(predicted, dtype=int)
        new = np.empty((len(x), len(self.columns_touched)))
        for cls in (0, 1):
            rows = target == cls
            pool = self.donors[cls]
            new[rows] = pool[rng.integers(0, len(pool), size=int(rows.sum()))]
        for j, col in enumerate(self.columns_touched):
            x[col] = new[:, j]
        return x

    def is_valid(self, x_new: pd.DataFrame) -> np.ndarray:
        ok = super().is_valid(x_new)
        if self.year_column is not None:
            year = pd.to_numeric(x_new[self.year_column], errors="coerce").to_numpy(dtype=float)
            ok &= (year >= self.valid[0]) & (year <= self.valid[1])
        return ok


def core_mrs(
    columns: Sequence[str],
    valid_years: tuple[int, int],
    year_column: str = "year",
    size_column: str = "apk_size",
    sdk_columns: tuple[str, str] = ("minSdkVersion", "targetSdkVersion"),
) -> list[MetamorphicRelation]:
    """MR1-MR4 for the metadata columns present in ``columns`` (fit them before use)."""
    present = set(columns)
    mrs: list[MetamorphicRelation] = []
    if year_column in present:
        mrs.append(YearShift(year_column, valid_years))
    if set(sdk_columns) <= present:
        mrs.append(SdkShift(*sdk_columns))
    if size_column in present:
        mrs.append(SizeScale(size_column))
    meta = [c for c in (year_column, *sdk_columns, size_column) if c in present]
    if meta:
        mrs.append(MetadataSwap(meta, year_column if year_column in present else None, valid_years))
    return mrs


__all__ = [
    "MetadataSwap",
    "SdkShift",
    "SizeScale",
    "YearShift",
    "changed_rows",
    "core_mrs",
]
