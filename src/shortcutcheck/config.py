"""Project paths and the Phase 1 configuration.

The project root is found by walking up from the working directory to the folder holding
``pyproject.toml``. Set ``SHORTCUTCHECK_ROOT`` to override it (tests use this).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    env = os.environ.get("SHORTCUTCHECK_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root (no pyproject.toml above cwd).")


@dataclass(frozen=True)
class Config:
    root: Path
    seeds: list[int]
    raw_dir: Path
    collection_year: int
    android_first_year: int
    dex_date_invalid_threshold: float
    metadata_columns: list[str]
    id_column: str
    dropped_features: list[str]
    rq3_primary: str
    rq12_splits: dict[str, list[str]]
    rq3_min_per_class_per_year: int
    rq3_candidates: dict[str, dict[str, Any]]
    split_seed: int
    rq12_ratios: list[float]
    rq3_val_fraction: float
    shuffled_label_tolerance: float
    shuffled_permutations: int
    mlflow_tracking_uri: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.root / "data" / "splits"

    @property
    def raw_root(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


_KNOWN = {
    "seeds",
    "raw_dir",
    "collection_year",
    "android_first_year",
    "dex_date_invalid_threshold",
    "metadata_columns",
    "id_column",
    "dropped_features",
    "rq3_primary",
    "rq12_splits",
    "rq3_min_per_class_per_year",
    "rq3_candidates",
    "split_seed",
    "rq12_ratios",
    "rq3_val_fraction",
    "shuffled_label_tolerance",
    "shuffled_permutations",
    "mlflow_tracking_uri",
}


def load_config(path: Path | None = None, root: Path | None = None) -> Config:
    root = root or project_root()
    path = path or root / "configs" / "phase1.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    ratios = [float(r) for r in raw["rq12_ratios"]]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"rq12_ratios must sum to 1, got {ratios}")
    if len(raw["seeds"]) != 5:
        raise ValueError("Rule 5: exactly five seeds are required.")
    return Config(
        root=root,
        seeds=[int(s) for s in raw["seeds"]],
        raw_dir=root / raw["raw_dir"],
        collection_year=int(raw["collection_year"]),
        android_first_year=int(raw["android_first_year"]),
        dex_date_invalid_threshold=float(raw["dex_date_invalid_threshold"]),
        metadata_columns=[str(c) for c in raw["metadata_columns"]],
        id_column=str(raw["id_column"]),
        dropped_features=[str(c) for c in raw.get("dropped_features", [])],
        rq3_primary=str(raw.get("rq3_primary", "")),
        rq12_splits={str(k): [str(x) for x in v] for k, v in raw.get("rq12_splits", {}).items()},
        rq3_min_per_class_per_year=int(raw["rq3_min_per_class_per_year"]),
        rq3_candidates={str(k): dict(v) for k, v in raw["rq3_candidates"].items()},
        split_seed=int(raw["split_seed"]),
        rq12_ratios=ratios,
        rq3_val_fraction=float(raw["rq3_val_fraction"]),
        shuffled_label_tolerance=float(raw["shuffled_label_tolerance"]),
        shuffled_permutations=int(raw["shuffled_permutations"]),
        mlflow_tracking_uri=str(raw["mlflow_tracking_uri"]),
        extra={k: v for k, v in raw.items() if k not in _KNOWN},
    )
