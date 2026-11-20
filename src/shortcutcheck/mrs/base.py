"""The metamorphic relation (MR) interface (roadmap step 2.1).

An MR states a change to an app's features that should not change a malware detector's
decision, because it does not change what the app does. Each MR:

* declares the columns it may change (``columns_touched``) and why the change should not
  matter (``justification``);
* is fitted on training inputs only (``fit``), which records the observed range of every
  touched column, so perturbed values never leave what the model saw in training;
* produces a perturbed copy of the inputs (``transform``);
* reports, per row, whether the perturbed values are inside the observed range (``is_valid``).

A row that the MR could not change (for example, MR3 at the edge of the size range) is
returned unchanged; the runner counts it as "not perturbed", not as a pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import pandas as pd

Role = Literal["core", "secondary"]


class NotFittedError(RuntimeError):
    """``transform`` or ``is_valid`` was called before ``fit``."""


class NotApplicableError(RuntimeError):
    """The MR cannot be applied to this data (the reason says why)."""


@dataclass(frozen=True)
class ObservedRange:
    """What one column looked like in the training inputs."""

    column: str
    low: float
    high: float
    values: tuple[float, ...]

    @classmethod
    def of(cls, series: pd.Series, column: str) -> ObservedRange:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            raise NotApplicableError(f"column {column!r} has no numeric values to fit on")
        values = tuple(float(v) for v in np.unique(clean.to_numpy(dtype=float)))
        return cls(column=column, low=values[0], high=values[-1], values=values)

    def contains(self, series: pd.Series) -> np.ndarray:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return np.asarray((x >= self.low) & (x <= self.high), dtype=bool)


class MetamorphicRelation(ABC):
    """Base class. Subclasses set the class attributes and implement ``_fit`` and ``_transform``."""

    mr_id: str
    name: str
    justification: str
    role: Role = "core"

    def __init__(self, columns_touched: tuple[str, ...]) -> None:
        if not columns_touched:
            raise ValueError("an MR must declare at least one column it may change")
        self._columns_touched = tuple(columns_touched)
        self._ranges: dict[str, ObservedRange] | None = None

    # ---- public interface -------------------------------------------------------------

    @property
    def columns_touched(self) -> tuple[str, ...]:
        """Every column this MR is allowed to change. Nothing else may differ."""
        return self._columns_touched

    @property
    def ranges(self) -> dict[str, ObservedRange]:
        if self._ranges is None:
            raise NotFittedError(f"{self.mr_id} has not been fitted")
        return self._ranges

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | None = None) -> Self:
        """Learn observed ranges from **training** inputs (never validation or test)."""
        missing = [c for c in self.columns_touched if c not in x_train.columns]
        if missing:
            raise NotApplicableError(f"{self.mr_id}: columns not in the data: {missing}")
        self._ranges = {c: ObservedRange.of(x_train[c], c) for c in self.columns_touched}
        self._fit(x_train, y_train)
        return self

    def transform(
        self,
        x: pd.DataFrame,
        *,
        rng: np.random.Generator,
        predicted: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Return a perturbed copy of ``x``. ``x`` itself is never modified.

        ``predicted`` holds the model's predicted class per row (0 goodware, 1 malware).
        Only MRs that need it (MR4) use it.
        """
        _ = self.ranges  # raises if not fitted
        out = self._transform(x.copy(), rng=rng, predicted=predicted)
        for col in self.columns_touched:
            out[col] = out[col].astype(x[col].dtype)
        return out

    def is_valid(self, x_new: pd.DataFrame) -> np.ndarray:
        """Per row: are all touched columns inside the observed training range?"""
        ok = np.ones(len(x_new), dtype=bool)
        for col, observed in self.ranges.items():
            ok &= observed.contains(x_new[col])
        return ok

    def describe(self) -> dict[str, str]:
        return {
            "id": self.mr_id,
            "name": self.name,
            "role": self.role,
            "columns": ", ".join(self.columns_touched),
            "justification": self.justification,
        }

    # ---- for subclasses -----------------------------------------------------------------

    def _fit(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:  # noqa: B027
        """Optional extra fitting (donor pools, valid value sets)."""

    @abstractmethod
    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        """Change only ``columns_touched`` of the copy ``x`` and return it."""


def changed_columns(before: pd.DataFrame, after: pd.DataFrame) -> set[str]:
    """Names of columns whose values differ anywhere (NaN equals NaN)."""
    if list(before.columns) != list(after.columns) or not before.index.equals(after.index):
        raise ValueError("frames must have the same columns and index")
    diff = before.ne(after) & ~(before.isna() & after.isna())
    return {str(c) for c in diff.columns[diff.any(axis=0)]}


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Per row: did any of ``columns`` change?"""
    b, a = before[list(columns)], after[list(columns)]
    diff = b.ne(a) & ~(b.isna() & a.isna())
    return np.asarray(diff.any(axis=1).to_numpy(), dtype=bool)
