"""Addendum B: splits that follow Dada's own training and test files exactly."""

from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from shortcutcheck.cli import app
from shortcutcheck.config import Config
from shortcutcheck.data.splits import read_index, split_paths
from shortcutcheck.evaluation.locked_test import access_log

runner = CliRunner()


def test_dada_split_keeps_files_in_their_roles_and_rebuilds(project: Config) -> None:
    args = ["data", "split-dada", "--name", "d-syn", "--train", "SYN_ACT14", "--test", "SYN_ACT17"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    train, val, test = (read_index(project, "d-syn", p) for p in ("train", "val", "test"))
    assert train and test and not val and not train & test
    frame = pd.read_csv(split_paths(project, "d-syn").test)
    assert set(frame["year"]) <= {2017, 1980}  # the test part is the 2017 file only
    manifest = json.loads((project.splits_dir / "d-syn" / "manifest.json").read_text())
    assert manifest["kind"] == "dada" and manifest["train_sets"] == ["SYN_ACT14"]
    assert not access_log(project)

    for f in (project.root / "data" / "processed").rglob("*.csv"):
        f.unlink()
    result = runner.invoke(app, ["data", "rebuild"])
    assert result.exit_code == 0 and "Rebuilt 'd-syn'" in result.output


def test_dada_split_without_a_test_file(project: Config) -> None:
    args = ["data", "split-dada", "--name", "d-train-only", "--train", "SYN_ACT14"]
    assert runner.invoke(app, args).exit_code == 0
    assert read_index(project, "d-train-only", "test") == set()
    assert read_index(project, "d-train-only", "train")
