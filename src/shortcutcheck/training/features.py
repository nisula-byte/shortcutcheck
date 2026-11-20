"""Turn a split frame into a numeric feature matrix for one feature set."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from shortcutcheck.data.dada import LABEL, feature_columns, split_feature_groups
from shortcutcheck.data.validity import to_year

FeatureSet = Literal["full", "behaviour"]


def numeric_features(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """All feature columns as floats; ``dex_date`` becomes a year; text columns are dropped."""
    out = {}
    for col in feature_columns(frame, id_column):
        if col == "dex_date":
            out[col] = to_year(frame[col])
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            out[col] = values.astype(float)
    return pd.DataFrame(out, index=frame.index)


def select(
    frame: pd.DataFrame, feature_set: FeatureSet, metadata_columns: list[str], id_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    x = numeric_features(frame, id_column)
    if feature_set == "behaviour":
        _, behaviour = split_feature_groups(list(x.columns), metadata_columns)
        x = x[behaviour]
    return x, frame[LABEL].astype(int)
