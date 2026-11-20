"""Steps 1.6-1.7: deduplication, frozen splits and the locked test set."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from shortcutcheck.config import Config
from shortcutcheck.data.dada import discover_sets, match_sets, read_set
from shortcutcheck.data.dedup import deduplicate
from shortcutcheck.data.splits import (
    LockedTestSetError,
    load_part,
    make_rq3_split,
    make_rq12_split,
    read_index,
    save_split,
)
from shortcutcheck.evaluation.locked_test import access_log, load_test

SRC = Path(__file__).resolve().parents[1] / "src" / "shortcutcheck"
H = [c * 64 for c in "abcdef"]


# --- 1.6 -----------------------------------------------------------------------------------


def test_dedup_rules() -> None:
    frame = pd.DataFrame(
        {
            "sha256": [H[0], H[0], H[1], H[1], H[2], H[3], H[4], H[5]],
            "f1": [1, 1, 0, 0, 5, 7, 7, 9],
            "label": [1, 1, 0, 1, 0, 0, 1, None],
        }
    )
    clean, report = deduplicate(frame)
    assert report.unlabelled_dropped == 1  # H[5]
    assert report.duplicate_rows_dropped == 1  # second H[0]
    assert report.conflicting_hashes_dropped == 2  # both H[1] rows
    assert set(clean["sha256"]) == {H[0], H[2], H[3], H[4]}
    assert clean["sha256"].is_unique
    # H[3] and H[4] share features (7) but disagree: flagged, not dropped.
    assert report.conflicting_feature_groups == 1
    assert report.rows_in_conflicting_feature_groups == 2


def _clean(cfg: Config, pattern: str) -> pd.DataFrame:
    sets = match_sets(discover_sets(cfg.raw_dir), pattern)
    frame = pd.concat([read_set(s, cfg.id_column) for s in sets], ignore_index=True)
    return deduplicate(frame, cfg.id_column)[0]


# --- 1.7 -----------------------------------------------------------------------------------


def test_rq12_split_has_zero_hash_overlap(project: Config) -> None:
    """The step 1.6 Check: no hash in more than one of train, validation and test."""
    frame = _clean(project, "SYN-DR-AG")
    save_split(project, "rq12", make_rq12_split(frame, 11, project.rq12_ratios), {})
    parts = {p: read_index(project, "rq12", p) for p in ("train", "val", "test")}
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
    total = sum(len(v) for v in parts.values())
    assert total == len(frame)
    assert len(parts["test"]) / total == pytest.approx(0.15, abs=0.01)


def test_rq12_split_is_stratified_and_reproducible(project: Config) -> None:
    frame = _clean(project, "SYN-DR-AG")
    a = make_rq12_split(frame, 11, project.rq12_ratios)
    b = make_rq12_split(frame, 11, project.rq12_ratios)
    pd.testing.assert_frame_equal(a, b)
    rates = a.groupby("part")["label"].mean()
    assert rates.max() - rates.min() < 0.02


def test_rq3_split_holds_out_late_and_removes_overlap(project: Config) -> None:
    early = _clean(project, "SYN_ACT14")
    late = _clean(project, "SYN_ACT17")
    late = pd.concat([late, early.head(3)], ignore_index=True)  # plant 3 overlapping apps
    split, removed = make_rq3_split(early, late, 11, 0.2, project.id_column)
    assert removed == 3
    test_hashes = set(split.loc[split["part"] == "test", "sha256"])
    assert not test_hashes & set(early["sha256"])
    assert set(split.loc[split["part"] != "test", "sha256"]) == set(early["sha256"])


def test_frozen_split_cannot_be_overwritten(project: Config) -> None:
    frame = _clean(project, "SYN-DR-AG")
    split = make_rq12_split(frame, 11, project.rq12_ratios)
    save_split(project, "frozen", split, {})
    with pytest.raises(FileExistsError):
        save_split(project, "frozen", split, {})


def test_training_loader_refuses_test(project: Config) -> None:
    frame = _clean(project, "SYN-DR-AG")
    save_split(project, "s", make_rq12_split(frame, 11, project.rq12_ratios), {})
    assert len(load_part(project, "s", "train")) > len(load_part(project, "s", "val"))
    with pytest.raises(LockedTestSetError):
        load_part(project, "s", "test")


def test_load_test_logs_every_call(project: Config) -> None:
    frame = _clean(project, "SYN-DR-AG")
    save_split(project, "s", make_rq12_split(frame, 11, project.rq12_ratios), {})
    with pytest.raises(ValueError):
        load_test(project, "s", purpose="  ")
    assert access_log(project) == []
    test = load_test(project, "s", purpose="unit test")
    load_test(project, "s", purpose="second look")
    log = access_log(project)
    assert [e["purpose"] for e in log] == ["unit test", "second look"]
    assert log[0]["rows"] == len(test) and log[0]["split"] == "s"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_only_evaluation_package_can_import_test_loader() -> None:
    """The step 1.7 Check: training code cannot import the test loader."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.parent.name == "evaluation":
            continue
        if any("locked_test" in name or name.endswith("load_test") for name in _imports(path)):
            offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == []
