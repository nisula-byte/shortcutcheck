"""Step 1.6: deduplicate by SHA-256 and flag identical feature rows with conflicting labels.

Rules:
* rows without a label are dropped (and counted);
* repeated hashes with one consistent label keep a single row;
* repeated hashes with conflicting labels are dropped entirely (the label is unreliable);
* identical feature vectors that carry both labels are *flagged*, not dropped, because two
  different apps can legitimately share a coarse DroidLysis vector. They are reported so the
  methodology chapter can quantify label noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from shortcutcheck.data.dada import LABEL, feature_columns


@dataclass(frozen=True)
class DedupReport:
    rows_in: int
    unlabelled_dropped: int
    duplicate_rows_dropped: int
    conflicting_hashes_dropped: int
    rows_out: int
    conflicting_feature_groups: int
    rows_in_conflicting_feature_groups: int


def deduplicate(frame: pd.DataFrame, id_column: str = "sha256") -> tuple[pd.DataFrame, DedupReport]:
    rows_in = len(frame)
    labelled = frame[frame[LABEL].notna()].copy()
    unlabelled = rows_in - len(labelled)
    labelled[LABEL] = labelled[LABEL].astype(int)

    labels_per_hash = labelled.groupby(id_column)[LABEL].nunique()
    conflicting = set(labels_per_hash[labels_per_hash > 1].index)
    consistent = labelled[~labelled[id_column].isin(conflicting)]
    deduped = consistent.drop_duplicates(subset=id_column, keep="first")

    features = feature_columns(deduped, id_column)
    if features:
        key = pd.util.hash_pandas_object(deduped[features], index=False)
        label_per_key = deduped.groupby(key)[LABEL].nunique()
        bad_keys = label_per_key[label_per_key > 1].index
        flagged = key.isin(bad_keys)
        groups, flagged_rows = len(bad_keys), int(flagged.sum())
    else:
        groups, flagged_rows = 0, 0

    report = DedupReport(
        rows_in=rows_in,
        unlabelled_dropped=unlabelled,
        duplicate_rows_dropped=len(consistent) - len(deduped),
        conflicting_hashes_dropped=int(labelled[id_column].isin(conflicting).sum()),
        rows_out=len(deduped),
        conflicting_feature_groups=groups,
        rows_in_conflicting_feature_groups=flagged_rows,
    )
    return deduped.reset_index(drop=True), report
