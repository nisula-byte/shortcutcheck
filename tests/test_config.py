from __future__ import annotations

from pathlib import Path

import pytest

from shortcutcheck.config import load_config

from .conftest import REPO, make_project


def test_repo_config_loads() -> None:
    cfg = load_config(root=REPO)
    assert len(cfg.seeds) == 5
    assert sum(cfg.rq12_ratios) == pytest.approx(1.0)
    assert "year" in cfg.metadata_columns
    assert "AndroCT-2014-vs-2017" in cfg.rq3_candidates
    assert cfg.rq3_primary in cfg.rq3_candidates
    assert cfg.dropped_features[0] == "dex_date"
    # Dada's debiasing columns are exact copies of others and must never reach a model.
    assert {"APK size", "Year", "Internet Permission"} <= set(cfg.dropped_features)
    assert not set(cfg.dropped_features) & {"year", "apk_size", "minSdkVersion"}
    # Exact names only: a glob like "DR-AG-*" would also pick up the DR-AG-C2_deb sets.
    for sets in cfg.rq12_splits.values():
        assert all("*" not in s for s in sets)


def test_config_rejects_wrong_seed_count(tmp_path: Path) -> None:
    root = make_project(tmp_path, seeds=[1, 2, 3])
    with pytest.raises(ValueError, match="five seeds"):
        load_config(root=root)
