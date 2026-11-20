"""Roadmap step 3.1 Check: two runs with the same seeds give identical numbers.

``compare_tables`` compares two result tables cell by cell. MLflow run IDs differ between runs
by design and are ignored; every other shared column must be exactly equal (NaN equals NaN).
There is no tolerance: on the reference platform (D15) reruns are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IGNORED = ("mlflow_run",)


@dataclass(frozen=True)
class TableComparison:
    n_rows: int
    identical_rows: int
    max_abs_diff: float
    columns: list[str]
    problems: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.problems and self.identical_rows == self.n_rows


def compare_tables(a: pd.DataFrame, b: pd.DataFrame) -> TableComparison:
    shared = [c for c in a.columns if c in b.columns and c not in IGNORED]
    problems = []
    if len(a) != len(b):
        problems.append(f"row counts differ: {len(a)} vs {len(b)}")
        return TableComparison(max(len(a), len(b)), 0, float("nan"), shared, problems)
    left = a[shared].reset_index(drop=True)
    right = b[shared].reset_index(drop=True)
    same = pd.Series(True, index=left.index)
    max_diff = 0.0
    for col in shared:
        x, y = left[col], right[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            eq = (xv == yv) | both_nan
            diff = np.where(both_nan, 0.0, np.abs(xv - yv))
            if diff.size and np.isfinite(diff).any():
                max_diff = max(max_diff, float(np.nanmax(diff)))
        else:
            eq = (x.astype(str) == y.astype(str)).to_numpy()
        same &= eq
    return TableComparison(len(left), int(same.sum()), max_diff, shared, problems)
