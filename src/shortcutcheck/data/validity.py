"""Step 1.5: is ``dex_date`` usable?

A ``dex_date`` is invalid if it is missing, zero, before Android existed, or after data
collection. Dada's README notes zero dates from the zip archive, and TIF/Chow et al. report
defaults such as 1980. If more than the configured share is invalid, MR1 drops ``dex_date``
and uses ``year`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def to_year(values: pd.Series) -> pd.Series:
    """Best-effort conversion of a dex_date column to a calendar year (float, NaN if unknown).

    Accepts plain years (2014), Unix timestamps in seconds or milliseconds, and date strings.
    Zero is treated as missing.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    years = np.full(len(values), np.nan)

    plain = numeric.between(1900, 2100).to_numpy()
    years[plain] = numeric[plain].to_numpy()

    ranges: tuple[tuple[float, float, Literal["s", "ms"]], ...] = (
        (1e8, 1e10, "s"),
        (1e11, 1e13, "ms"),
    )
    for low, high, unit in ranges:
        mask = numeric.between(low, high).to_numpy()
        if mask.any():
            stamps = pd.to_datetime(numeric[mask], unit=unit, errors="coerce")
            years[mask] = stamps.dt.year.to_numpy(dtype=float)

    text = (numeric.isna() & values.notna()).to_numpy()
    if text.any():
        parsed = pd.to_datetime(values[text].astype(str), errors="coerce", format="mixed")
        years[text] = parsed.dt.year.to_numpy(dtype=float)
    return pd.Series(years, index=values.index, dtype=float)


@dataclass(frozen=True)
class ValidityReport:
    total: int
    missing_or_zero: int
    before_android: int
    after_collection: int
    exactly_1980: int

    @property
    def invalid(self) -> int:
        return self.missing_or_zero + self.before_android + self.after_collection

    @property
    def invalid_share(self) -> float:
        return 0.0 if self.total == 0 else self.invalid / self.total


def check_dex_date(values: pd.Series, first_year: int, last_year: int) -> ValidityReport:
    years = to_year(values)
    return ValidityReport(
        total=len(values),
        missing_or_zero=int(years.isna().sum()),
        before_android=int((years < first_year).sum()),
        after_collection=int((years > last_year).sum()),
        exactly_1980=int((years == 1980).sum()),
    )


def decision(report: ValidityReport, threshold: float) -> str:
    if report.invalid_share > threshold:
        return (
            f"DROP dex_date: {report.invalid_share:.1%} invalid (> {threshold:.0%}). "
            "MR1 uses year alone."
        )
    return (
        f"KEEP dex_date: {report.invalid_share:.1%} invalid (<= {threshold:.0%}). "
        "Invalid rows are excluded from MR1 perturbation targets."
    )
