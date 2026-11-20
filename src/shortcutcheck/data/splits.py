"""Step 1.7: frozen splits.

* RQ1/RQ2: stratified 70/15/15 with a fixed seed.
* RQ3: early-year data split into train/validation; late-year data held out entirely as test.
  Any late-year app whose hash also appears in the early data is removed from the test set.

Each split is saved twice:
* ``data/splits/<name>/{train,val,test}.txt`` hold sorted hashes plus ``manifest.json``.
  These are small and are committed to git.
* ``data/processed/<name>/trainval.csv.gz`` and ``test.csv.gz`` hold features. Training code
  reads only ``trainval``; ``test`` is read solely by
  :func:`shortcutcheck.evaluation.locked_test.load_test`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from shortcutcheck.config import Config
from shortcutcheck.data.dada import LABEL
from shortcutcheck.data.safe_io import write_frame, write_text

PART = "part"
TRAINING_PARTS = ("train", "val")


class LockedTestSetError(PermissionError):
    """Raised when training code asks for the locked test partition."""


@dataclass(frozen=True)
class SplitPaths:
    index_dir: Path
    data_dir: Path

    @property
    def trainval(self) -> Path:
        return self.data_dir / "trainval.csv.gz"

    @property
    def test(self) -> Path:
        return self.data_dir / "test.csv.gz"


def split_paths(cfg: Config, name: str) -> SplitPaths:
    return SplitPaths(cfg.splits_dir / name, cfg.processed_dir / name)


def make_rq12_split(frame: pd.DataFrame, seed: int, ratios: list[float]) -> pd.DataFrame:
    train_r, val_r, test_r = ratios
    rest, test = train_test_split(frame, test_size=test_r, stratify=frame[LABEL], random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_r / (train_r + val_r), stratify=rest[LABEL], random_state=seed
    )
    split: pd.DataFrame = pd.concat(
        [
            train.assign(**{PART: "train"}),
            val.assign(**{PART: "val"}),
            test.assign(**{PART: "test"}),
        ],
        ignore_index=True,
    )
    return split


def make_rq3_split(
    early: pd.DataFrame, late: pd.DataFrame, seed: int, val_fraction: float, id_column: str
) -> tuple[pd.DataFrame, int]:
    """Returns the split frame and the number of late apps removed for overlapping hashes."""
    overlap = late[id_column].isin(set(early[id_column]))
    late = late[~overlap]
    train, val = train_test_split(
        early, test_size=val_fraction, stratify=early[LABEL], random_state=seed
    )
    split = pd.concat(
        [
            train.assign(**{PART: "train"}),
            val.assign(**{PART: "val"}),
            late.assign(**{PART: "test"}),
        ],
        ignore_index=True,
    )
    return split, int(overlap.sum())


def save_split(
    cfg: Config, name: str, split: pd.DataFrame, manifest_extra: dict[str, object]
) -> SplitPaths:
    paths = split_paths(cfg, name)
    if paths.index_dir.exists() and any(paths.index_dir.iterdir()):
        raise FileExistsError(
            f"Split '{name}' is already frozen at {paths.index_dir}. Frozen splits are never "
            "regenerated; create a new split name and record why."
        )
    id_col = cfg.id_column
    counts: dict[str, dict[str, int]] = {}
    for part in ("train", "val", "test"):
        rows = split[split[PART] == part]
        write_text(
            paths.index_dir / f"{part}.txt",
            "\n".join(sorted(rows[id_col])) + "\n",
            cfg.raw_root,
        )
        counts[part] = {
            "rows": len(rows),
            "goodware": int((rows[LABEL] == 0).sum()),
            "malware": int((rows[LABEL] == 1).sum()),
        }
    manifest = {
        "name": name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": counts,
        **manifest_extra,
    }
    write_text(paths.index_dir / "manifest.json", json.dumps(manifest, indent=2), cfg.raw_root)
    return write_processed(cfg, name, split)


def write_processed(cfg: Config, name: str, split: pd.DataFrame) -> SplitPaths:
    """(Re)write the processed feature files of a split. Index files are not touched."""
    paths = split_paths(cfg, name)
    write_frame(split[split[PART].isin(TRAINING_PARTS)], paths.trainval, cfg.raw_root)
    write_frame(split[split[PART] == "test"].drop(columns=[PART]), paths.test, cfg.raw_root)
    return paths


def verify_against_index(
    cfg: Config, name: str, split: pd.DataFrame, sources: dict[str, str]
) -> None:
    """Raise unless ``split`` reproduces the frozen split exactly (sources, apps, counts)."""
    paths = split_paths(cfg, name)
    manifest = json.loads((paths.index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sources") != sources:
        raise ValueError(
            f"Split '{name}': source files differ from the manifest "
            f"(expected {manifest.get('sources')}, found {sources})."
        )
    for part in ("train", "val", "test"):
        rows = split[split[PART] == part]
        if set(rows[cfg.id_column]) != read_index(cfg, name, part):
            raise ValueError(f"Split '{name}/{part}': apps differ from the frozen index file.")
        expected = manifest["counts"][part]
        found = {
            "rows": len(rows),
            "goodware": int((rows[LABEL] == 0).sum()),
            "malware": int((rows[LABEL] == 1).sum()),
        }
        if found != expected:
            raise ValueError(f"Split '{name}/{part}': counts {found} != manifest {expected}.")


def read_index(cfg: Config, name: str, part: str) -> set[str]:
    path = split_paths(cfg, name).index_dir / f"{part}.txt"
    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def load_part(cfg: Config, name: str, part: str) -> pd.DataFrame:
    """Training-side loader. Only 'train' and 'val' are allowed."""
    if part not in TRAINING_PARTS:
        raise LockedTestSetError(
            "Test data is locked. Use shortcutcheck.evaluation.locked_test.load_test(), "
            "once per experiment, at the end."
        )
    frame = pd.read_csv(
        split_paths(cfg, name).trainval, low_memory=False, dtype={cfg.id_column: str}
    )
    selected = frame[frame[PART] == part].drop(columns=[PART]).reset_index(drop=True)
    if set(selected[cfg.id_column]) != read_index(cfg, name, part):
        raise ValueError(f"Split '{name}/{part}' does not match its frozen index file.")
    return selected
