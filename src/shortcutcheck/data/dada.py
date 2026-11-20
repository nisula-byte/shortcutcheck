"""Read Dada datasets (feature CSVs and labels) from the read-only raw folder.

Dada ships, per dataset, ``<name>.characteristics.csv`` (DroidLysis features keyed by
SHA-256) and, for mixed sets, ``<name>.goodmal.csv`` (goodware = 0, malware = 1). The exact
label-file layout is confirmed in step 1.4, so the label reader accepts either a headed or a
headerless two-column file.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CHAR_SUFFIX = ".characteristics.csv"
LABEL_SUFFIX = ".goodmal.csv"
LABEL = "label"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_LABEL_COLUMNS = ("malware", "goodmal", "label")
_LABEL_WORDS = {"0": 0, "1": 1, "goodware": 0, "malware": 1, "benign": 0, "malicious": 1}


@dataclass(frozen=True)
class DadaSet:
    name: str
    features_path: Path
    labels_path: Path | None


def discover_sets(raw_dir: Path) -> list[DadaSet]:
    """Every ``*.characteristics.csv`` under ``raw_dir``, paired with its label file."""
    sets = []
    for path in sorted(raw_dir.rglob(f"*{CHAR_SUFFIX}")):
        name = path.name[: -len(CHAR_SUFFIX)]
        labels = path.with_name(name + LABEL_SUFFIX)
        sets.append(DadaSet(name, path, labels if labels.exists() else None))
    return sets


def match_sets(sets: list[DadaSet], pattern: str) -> list[DadaSet]:
    return [s for s in sets if fnmatch.fnmatch(s.name, pattern)]


def read_labels(path: Path, id_column: str = "sha256") -> pd.DataFrame:
    """Return a frame with columns ``[id_column, label]``; label is 0 (goodware) or 1."""
    raw = pd.read_csv(path, header=None, dtype=str)
    if raw.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns (hash, label).")
    first = raw.iloc[0].astype(str).str.strip()
    has_header = not any(_HEX64.match(v) for v in first)
    named: dict[str, int] = {}
    if has_header:
        named = {name.lower(): idx for idx, name in enumerate(first)}
        raw = raw.iloc[1:].reset_index(drop=True)
    hash_col = next(
        (c for c in raw.columns if raw[c].astype(str).str.strip().str.match(_HEX64).all()),
        None,
    )
    if hash_col is None:
        raise ValueError(f"{path}: no column of 64-character hex hashes found.")
    # Dada v3.1 label files are "sha256,malware,dataset"; pick the label column by name when
    # there is a header, otherwise fall back to the first non-hash column.
    by_name = next((named[n] for n in _LABEL_COLUMNS if n in named), None)
    label_col = (
        raw.columns[by_name]
        if by_name is not None
        else next(c for c in raw.columns if c != hash_col)
    )
    values = raw[label_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(values) - set(_LABEL_WORDS))
    if unknown:
        raise ValueError(f"{path}: unrecognised label values {unknown[:5]}")
    return pd.DataFrame(
        {
            id_column: raw[hash_col].astype(str).str.strip().str.lower(),
            LABEL: values.map(_LABEL_WORDS).astype(int),
        }
    )


def read_set(dada_set: DadaSet, id_column: str = "sha256") -> pd.DataFrame:
    """Features joined with labels (``label`` is missing if the set has no label file)."""
    features = pd.read_csv(dada_set.features_path, low_memory=False)
    if id_column not in features.columns:
        raise ValueError(f"{dada_set.features_path}: no '{id_column}' column.")
    features[id_column] = features[id_column].astype(str).str.strip().str.lower()
    if dada_set.labels_path is None:
        return features
    labels = read_labels(dada_set.labels_path, id_column)
    return features.merge(labels, on=id_column, how="left", validate="many_to_many")


def feature_columns(frame: pd.DataFrame, id_column: str = "sha256") -> list[str]:
    return [c for c in frame.columns if c not in (id_column, LABEL)]


def split_feature_groups(
    columns: list[str], metadata_columns: list[str]
) -> tuple[list[str], list[str]]:
    """(metadata, behaviour) column lists, preserving order."""
    meta = set(metadata_columns)
    return [c for c in columns if c in meta], [c for c in columns if c not in meta]
