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


def test_config_rejects_wrong_seed_count(tmp_path: Path) -> None:
    root = make_project(tmp_path, seeds="[1, 2, 3]")
    with pytest.raises(ValueError, match="five seeds"):
        load_config(root=root)
