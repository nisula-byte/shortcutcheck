"""Roadmap step 3.1: one YAML config per experiment, and identical reruns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from shortcutcheck.cli import app
from shortcutcheck.config import Config, load_config
from shortcutcheck.data.splits import read_index
from shortcutcheck.evaluation.experiment import run_experiment, stratified_subsample
from shortcutcheck.evaluation.locked_test import access_log
from shortcutcheck.experiments.compare import compare_tables
from shortcutcheck.experiments.spec import ExperimentConfigError, load_experiment
from shortcutcheck.pilot import pilot_experiment
from shortcutcheck.training.baselines import build_model

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
runner = CliRunner()


def repo_cfg() -> Config:
    return load_config(root=REPO)


# ---- configs ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["pilot", "rq1", "rq2", "rq3"])
def test_every_experiment_config_is_valid(name: str) -> None:
    exp = load_experiment(EXPERIMENTS / f"{name}.yaml", repo_cfg())
    assert exp.experiment == name and exp.seeds == (11, 22, 33, 44, 55)


def test_pilot_config_describes_the_step_2_7_pilot() -> None:
    cfg = repo_cfg()
    from_yaml = load_experiment(EXPERIMENTS / "pilot.yaml", cfg)
    in_code = pilot_experiment(cfg, "rq12-DR-AG")
    for attr in ("seeds", "algorithms", "feature_sets", "mrs", "realism", "mr_sample"):
        assert getattr(from_yaml, attr) == getattr(in_code, attr), attr
    (run,) = from_yaml.runs
    assert run.train == "rq12-DR-AG" and [(t.split, t.partition) for t in run.targets] == [
        ("rq12-DR-AG", "val")
    ]
    assert from_yaml.test_splits == ()  # the 3.1 rerun check opens no test data


def test_rq2_scores_paired_models_on_identical_inputs() -> None:
    """Within a family, every model's h2 inputs exclude the same training apps."""
    exp = load_experiment(EXPERIMENTS / "rq2.yaml", repo_cfg())
    for family in ("DR-AG", "VS-AG"):
        runs = [r for r in exp.runs if r.train.startswith(f"rq12-{family}")]
        for role in ("h2-deb", "h2-biased"):
            excluded = {
                frozenset({r.train, *t.exclude_train_of})
                for r in runs
                for t in r.targets
                if t.role == role
            }
            assert excluded == {frozenset({f"rq12-{family}", f"rq12-{family}_deb"})}, role


def test_rq3_hyperparameter_settings_all_build() -> None:
    exp = load_experiment(EXPERIMENTS / "rq3.yaml", repo_cfg())
    hp = exp.pool["hyperparameters"]
    assert all(len(v) == 6 and v[0] == {} for v in hp.values())
    assert exp.pool["skew"]["settings_per_level"] == [6, 2, 2, 2]


def _write(tmp_path: Path, **changes: object) -> Path:
    raw = yaml.safe_load((EXPERIMENTS / "rq1.yaml").read_text())
    raw.update(changes)
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": [1, 2, 3, 4]},
        {"seeds": [1, 1, 2, 3, 4]},
        {"algorithms": ["lr", "xgb"]},
        {"mrs": ["MR9"]},
        {"engine": "magic"},
        {
            "runs": [
                {
                    "name": "a",
                    "train": "nope",
                    "evaluate": [{"split": "rq12-DR-AG", "partition": "test"}],
                }
            ]
        },
        {
            "runs": [
                {
                    "name": "a",
                    "train": "rq12-DR-AG",
                    "arm": "size-matched",
                    "evaluate": [{"split": "rq12-DR-AG", "partition": "test"}],
                }
            ]
        },
        {
            "runs": [
                {
                    "name": "a",
                    "train": "rq12-DR-AG",
                    "evaluate": [{"split": "rq12-DR-AG", "partition": "train"}],
                }
            ]
        },
        {
            "runs": [
                {
                    "name": "a",
                    "train": "rq12-DR-AG",
                    "evaluate": [
                        {"split": "rq12-DR-AG", "partition": "val"},
                        {"split": "rq12-DR-AG", "partition": "test"},
                    ],
                }
            ]
        },
    ],
)
def test_invalid_configs_fail_before_any_data_is_read(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ExperimentConfigError):
        load_experiment(_write(tmp_path, **changes), repo_cfg())


def test_unknown_hyperparameter_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXPERIMENTS / "rq3.yaml").read_text())
    raw["pool"]["hyperparameters"]["lr"][1] = {"Cc": 0.01}
    path = tmp_path / "rq3.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment(path, repo_cfg())


def test_build_model_params() -> None:
    assert repr(build_model("lr", 1, params={})) == repr(build_model("lr", 1))
    assert build_model("svm", 1, params={"C": 0.1}).get_params()["model__estimator__C"] == 0.1
    assert build_model("rf", 1, params={"max_depth": 4}).get_params()["model__max_depth"] == 4
    with pytest.raises(ValueError):
        build_model("mlp", 1, params={"layers": 3})


# ---- helpers -----------------------------------------------------------------------------


def test_stratified_subsample_keeps_shares_and_is_seeded() -> None:
    y = pd.Series([1] * 300 + [0] * 700)
    a = stratified_subsample(y, 100, seed=5)
    assert len(a) == 100 and y.iloc[a].sum() == 30 and len(set(a)) == 100
    assert np.array_equal(a, stratified_subsample(y, 100, seed=5))
    assert not np.array_equal(a, stratified_subsample(y, 100, seed=6))
    assert np.array_equal(stratified_subsample(y, 5000, seed=5), np.arange(1000))


def test_compare_tables() -> None:
    a = pd.DataFrame({"k": ["x", "y"], "rate": [0.1, np.nan], "mlflow_run": ["r1", "r2"]})
    b = a.assign(mlflow_run=["r3", "r4"])
    assert compare_tables(a, b).identical  # run IDs ignored, NaN equals NaN
    c = compare_tables(a, b.assign(rate=[0.1 + 1e-15, np.nan]))
    assert not c.identical and c.identical_rows == 1 and c.max_abs_diff > 0
    assert not compare_tables(a, b.iloc[:1]).identical


# ---- engine on synthetic data -------------------------------------------------------------


def _synthetic_splits(project: Config) -> None:
    assert (
        runner.invoke(
            app, ["data", "split-rq12", "--name", "rq12-syn", "--sets", "SYN-DR-AG"]
        ).exit_code
        == 0
    )
    # A second split of the same apps with another seed: its test partition overlaps
    # rq12-syn's training partition, so exclusion has something to remove.
    path = project.root / "configs" / "phase1.yaml"
    raw = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**raw, "split_seed": 99}))
    assert (
        runner.invoke(
            app, ["data", "split-rq12", "--name", "rq12-syn-b", "--sets", "SYN-DR-AG"]
        ).exit_code
        == 0
    )
    path.write_text(yaml.safe_dump(raw))


def _small_experiment(project: Config, mr_sample: int | None = 150) -> Path:
    raw = {
        "experiment": "toy",
        "step": "3.1-test",
        "engine": "models",
        "mlflow_experiment": "toy",
        "seeds": [11, 22, 33, 44, 55],
        "algorithms": ["lr", "rf"],
        "feature_sets": ["full", "behaviour"],
        "mrs": ["MR1", "MR3", "MR4"],
        "realism": True,
        "mr_sample": mr_sample,
        "runs": [
            {
                "name": "syn",
                "train": "rq12-syn",
                "evaluate": [
                    {"split": "rq12-syn", "partition": "test", "role": "own"},
                    {"split": "rq12-syn-b", "partition": "test", "role": "other"},
                ],
            },
            {
                "name": "syn-small",
                "train": "rq12-syn",
                "arm": "size-matched",
                "size_like": "rq12-syn-b",
                "feature_sets": ["full"],
                "evaluate": [{"split": "rq12-syn", "partition": "val", "role": "own"}],
            },
        ],
    }
    path = project.root / "configs" / "experiments" / "toy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_engine_on_synthetic_data_and_identical_rerun(project: Config) -> None:
    _synthetic_splits(project)
    exp = load_experiment(_small_experiment(project), project)
    first = run_experiment(project, exp, "abc")

    # Each test partition opened exactly once, with the purpose logged.
    log = access_log(project)
    assert sorted(e["split"] for e in log) == ["rq12-syn", "rq12-syn-b"]
    assert all("toy (roadmap step 3.1-test)" in str(e["purpose"]) for e in log)

    # No model scored on an app it was trained on.
    inputs = first.inputs.set_index("role")
    overlap = read_index(project, "rq12-syn", "train") & read_index(project, "rq12-syn-b", "test")
    other = first.inputs[first.inputs["role"] == "other"].iloc[0]
    assert overlap and other["n_excluded"] == len(overlap)
    assert other["n_inputs"] == len(read_index(project, "rq12-syn-b", "test")) - len(overlap)
    assert (first.inputs["n_mr_sample"] <= 150).all()
    assert set(inputs.index) == {"own", "other"}

    # Behaviour-only models stay exactly 0 on metadata MRs through the engine too.
    beh = first.rows[(first.rows["feature_set"] == "behaviour")]
    assert (beh["n_violations"] == 0).all()
    assert set(first.rows["run"]) == {"syn", "syn-small"}
    assert set(first.rows["arm"]) == {"published", "size-matched"}

    # The step 3.1 Check: a second run with the same seeds is identical.
    second = run_experiment(project, exp, "abc")
    for a, b in ((first.rows, second.rows), (first.detection, second.detection)):
        c = compare_tables(a, b)
        assert c.identical, (c.identical_rows, c.n_rows)


def test_experiment_cli_run_and_compare(project: Config) -> None:
    _synthetic_splits(project)
    path = _small_experiment(project, mr_sample=None)
    raw = yaml.safe_load(path.read_text())
    raw["runs"] = raw["runs"][1:]  # validation only: no test access
    path.write_text(yaml.safe_dump(raw))
    for out in ("a", "b"):
        result = runner.invoke(
            app, ["experiment", "run", str(path), "--out", str(project.root / out)]
        )
        assert result.exit_code == 0, result.output
    assert (
        "Test partitions opened (once each, logged in `logs/test_access.jsonl`): none"
        in (project.root / "a" / "run.md").read_text()
    )
    result = runner.invoke(
        app, ["experiment", "compare", str(project.root / "a"), str(project.root / "b")]
    )
    assert result.exit_code == 0 and "Verdict: **IDENTICAL**" in result.output
    rows = pd.read_csv(project.root / "a" / "rows.csv")
    rows.loc[0, "rate"] = 0.123456
    rows.to_csv(project.root / "a" / "rows.csv", index=False)
    result = runner.invoke(
        app, ["experiment", "compare", str(project.root / "a"), str(project.root / "b")]
    )
    assert result.exit_code == 1 and "DIFFERENT" in result.output
    assert not (project.root / "logs" / "test_access.jsonl").exists()
