"""A locked, test-only holdout window (Phase A, step 1: fresh 2015-16 data for H5)."""

from __future__ import annotations

import json

import pandas as pd
import yaml
from typer.testing import CliRunner

from shortcutcheck.cli import app
from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part, read_index, split_paths
from shortcutcheck.evaluation.locked_test import access_log, load_test

runner = CliRunner()


def _candidate_leaving_2017_free(project: Config) -> None:
    path = project.root / "configs" / "phase1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["rq3_candidates"]["SYN-early-only"] = {
        "labelled": ["SYN_ACT1*"],
        "early_years": [2014],
        "late_years": [2016],
    }
    path.write_text(yaml.safe_dump(raw))


def test_holdout_is_test_only_disjoint_and_rebuildable(project: Config) -> None:
    _candidate_leaving_2017_free(project)
    args = [
        "data",
        "split-holdout",
        "--candidate",
        "SYN-early-only",
        "--years",
        "2017",
        "--name",
        "hold-2017",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert read_index(project, "hold-2017", "train") == set()
    assert read_index(project, "hold-2017", "val") == set()
    held = read_index(project, "hold-2017", "test")
    assert len(held) > 100
    assert load_part(project, "hold-2017", "val").empty  # nothing readable outside load_test
    assert not access_log(project)  # freezing it opened nothing

    manifest = json.loads((project.splits_dir / "hold-2017" / "manifest.json").read_text())
    assert manifest["kind"] == "holdout" and manifest["years"] == [2017]

    # Disjoint from every app of the primary split built from the same candidate.
    assert (
        runner.invoke(
            app, ["data", "split-rq3", "--candidate", "SYN-early-only", "--name", "rq3-early"]
        ).exit_code
        == 0
    )
    used = set().union(*(read_index(project, "rq3-early", p) for p in ("train", "val", "test")))
    assert not held & used

    frame = load_test(project, "hold-2017", purpose="test")
    assert set(frame["year"]) == {2017} and len(frame) == len(held)
    assert len(access_log(project)) == 1

    # A second machine rebuilds it and verifies it against the frozen index.
    for f in (project.root / "data" / "processed").rglob("*.csv"):
        f.unlink()
    result = runner.invoke(app, ["data", "rebuild"])
    assert result.exit_code == 0 and "Rebuilt 'hold-2017'" in result.output
    assert len(pd.read_csv(split_paths(project, "hold-2017").test)) == len(held)


def test_holdout_cannot_be_refrozen(project: Config) -> None:
    _candidate_leaving_2017_free(project)
    args = [
        "data",
        "split-holdout",
        "--candidate",
        "SYN-early-only",
        "--years",
        "2017",
        "--name",
        "hold-2017",
    ]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code != 0  # frozen splits are never regenerated
