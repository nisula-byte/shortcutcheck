"""One YAML config per experiment (roadmap step 3.1).

An experiment config says everything a run needs: seeds, algorithms, feature sets, MRs, which
model is trained on which split, and which inputs each model is evaluated on. Nothing about an
experiment is hard-coded elsewhere. ``load_experiment`` validates a config before any data is
read, so a typo fails at once instead of after hours of training.

Two engines exist:

* ``models`` (pilot, RQ1, RQ2): train models on frozen splits, run MRs on evaluation inputs.
* ``pool`` (RQ3): the 2012-14 model pool with skew injection and label-free decay signals.
  Its settings are validated here; the pool itself is built in steps 3.5-3.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from shortcutcheck.config import Config
from shortcutcheck.training.baselines import ALL_ALGORITHMS, build_model

Partition = Literal["val", "test"]
Arm = Literal["published", "size-matched"]
MR_IDS = ("MR1", "MR2", "MR3", "MR4", "MR5", "MR6")
FEATURE_SET_NAMES = ("full", "behaviour")


class ExperimentConfigError(ValueError):
    """The experiment config is invalid."""


@dataclass(frozen=True)
class Target:
    """Inputs a model is evaluated on.

    MRs and the realism check are fitted on ``split``'s training partition (they describe the
    input population). Apps in the training partition of the model's own split, or of any split
    in ``exclude_train_of``, are removed so no model is scored on apps it was trained on.
    """

    split: str
    partition: Partition
    role: str
    exclude_train_of: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRun:
    name: str
    train: str
    arm: Arm
    targets: tuple[Target, ...]
    size_like: str | None = None
    feature_sets: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Experiment:
    path: Path
    experiment: str
    step: str
    engine: Literal["models", "pool"]
    mlflow_experiment: str
    seeds: tuple[int, ...]
    algorithms: tuple[str, ...]
    feature_sets: tuple[str, ...]
    mrs: tuple[str, ...]
    realism: bool
    mr_sample: int | None
    sample_seed: int
    runs: tuple[ModelRun, ...]
    uses: tuple[str, ...] = ()
    pool: dict[str, Any] = field(default_factory=dict)

    def feature_sets_for(self, run: ModelRun) -> tuple[str, ...]:
        return run.feature_sets or self.feature_sets

    @property
    def test_splits(self) -> tuple[str, ...]:
        """Splits whose test partition this experiment opens (each once, via load_test)."""
        seen: dict[str, None] = {}
        for run in self.runs:
            for t in run.targets:
                if t.partition == "test":
                    seen[t.split] = None
        if self.engine == "pool":
            seen[str(self.pool["split"])] = None
        return tuple(seen)


def _need(raw: dict[str, Any], key: str, where: str) -> Any:
    if key not in raw:
        raise ExperimentConfigError(f"{where}: missing '{key}'.")
    return raw[key]


def _subset(values: list[Any], allowed: tuple[str, ...], what: str) -> tuple[str, ...]:
    bad = [v for v in values if v not in allowed]
    if bad or not values:
        raise ExperimentConfigError(f"Unknown or empty {what}: {bad or values}; allowed {allowed}.")
    if len(set(values)) != len(values):
        raise ExperimentConfigError(f"Duplicate {what}: {values}.")
    return tuple(str(v) for v in values)


def _split(cfg: Config, name: str, where: str) -> str:
    known = set(cfg.rq12_splits) | {f"rq3-{cfg.rq3_primary}"}
    on_disk = (cfg.splits_dir / name / "manifest.json").exists()
    if name not in known and not on_disk:
        raise ExperimentConfigError(f"{where}: unknown split '{name}'.")
    return name


def _targets(cfg: Config, raw: list[dict[str, Any]], where: str) -> tuple[Target, ...]:
    out = []
    for i, t in enumerate(raw):
        here = f"{where}, evaluate[{i}]"
        partition = _need(t, "partition", here)
        if partition not in ("val", "test"):
            raise ExperimentConfigError(f"{here}: partition must be 'val' or 'test'.")
        out.append(
            Target(
                split=_split(cfg, str(_need(t, "split", here)), here),
                partition=partition,
                role=str(t.get("role", "own")),
                exclude_train_of=tuple(
                    _split(cfg, str(s), here) for s in t.get("exclude_train_of", [])
                ),
            )
        )
    roles = [t.role for t in out]
    if not out or len(set(roles)) != len(roles):
        raise ExperimentConfigError(f"{where}: needs at least one target, with distinct roles.")
    return tuple(out)


def _runs(cfg: Config, raw: list[dict[str, Any]]) -> tuple[ModelRun, ...]:
    runs = []
    for i, r in enumerate(raw):
        where = f"runs[{i}]"
        arm = r.get("arm", "published")
        if arm not in ("published", "size-matched"):
            raise ExperimentConfigError(f"{where}: arm must be 'published' or 'size-matched'.")
        size_like = r.get("size_like")
        if (arm == "size-matched") != (size_like is not None):
            raise ExperimentConfigError(
                f"{where}: 'size_like' is required for, and only for, " "the size-matched arm."
            )
        fs = r.get("feature_sets")
        runs.append(
            ModelRun(
                name=str(_need(r, "name", where)),
                train=_split(cfg, str(_need(r, "train", where)), where),
                arm=arm,
                size_like=_split(cfg, str(size_like), where) if size_like else None,
                targets=_targets(cfg, _need(r, "evaluate", where), where),
                feature_sets=_subset(fs, FEATURE_SET_NAMES, "feature sets") if fs else None,
            )
        )
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ExperimentConfigError(f"Run names must be distinct: {names}.")
    return tuple(runs)


def _check_pool(raw: dict[str, Any], algorithms: tuple[str, ...], seed: int) -> dict[str, Any]:
    """Validate the RQ3 pool settings: skew levels and buildable hyperparameter settings."""
    for key in ("split", "skew", "hyperparameters", "signals", "analysis"):
        _need(raw, key, "pool")
    levels = raw["skew"]["levels"]
    if levels[0] != 0 or sorted(levels) != levels or len(set(levels)) != len(levels):
        raise ExperimentConfigError("pool.skew.levels must start at 0 and increase.")
    per_level = raw["skew"]["settings_per_level"]
    if len(per_level) != len(levels):
        raise ExperimentConfigError("pool.skew.settings_per_level needs one entry per level.")
    hp = raw["hyperparameters"]
    if set(hp) != set(algorithms):
        raise ExperimentConfigError("pool.hyperparameters needs exactly one entry per algorithm.")
    for algorithm, settings in hp.items():
        if len(settings) < max(per_level) or settings[0] != {}:
            raise ExperimentConfigError(
                f"pool.hyperparameters.{algorithm}: setting 0 must be {{}} (the RQ1 defaults) "
                f"and there must be at least {max(per_level)} settings."
            )
        for params in settings:
            build_model(algorithm, seed, params=params)  # raises on an unknown parameter
    return dict(raw)


def load_experiment(path: Path, cfg: Config) -> Experiment:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    engine = _need(raw, "engine", str(path))
    if engine not in ("models", "pool"):
        raise ExperimentConfigError("engine must be 'models' or 'pool'.")
    seeds = tuple(int(s) for s in _need(raw, "seeds", str(path)))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ExperimentConfigError("Rule 5: exactly five distinct seeds.")
    algorithms = _subset(_need(raw, "algorithms", str(path)), ALL_ALGORITHMS, "algorithms")
    mr_sample = raw.get("mr_sample")
    if mr_sample is not None and int(mr_sample) < 1:
        raise ExperimentConfigError("mr_sample must be positive or null (all inputs).")
    exp = Experiment(
        path=Path(path),
        experiment=str(_need(raw, "experiment", str(path))),
        step=str(_need(raw, "step", str(path))),
        engine=engine,
        mlflow_experiment=str(_need(raw, "mlflow_experiment", str(path))),
        seeds=seeds,
        algorithms=algorithms,
        feature_sets=_subset(raw.get("feature_sets", ["full"]), FEATURE_SET_NAMES, "feature sets"),
        mrs=_subset(raw.get("mrs", list(MR_IDS)), MR_IDS, "MRs"),
        realism=bool(raw.get("realism", True)),
        mr_sample=None if mr_sample is None else int(mr_sample),
        sample_seed=int(raw.get("sample_seed", cfg.split_seed)),
        runs=_runs(cfg, raw.get("runs", [])) if engine == "models" else (),
        uses=tuple(str(u) for u in raw.get("uses", [])),
        pool=_check_pool(raw["pool"], algorithms, seeds[0]) if engine == "pool" else {},
    )
    if engine == "models" and not exp.runs:
        raise ExperimentConfigError("A 'models' experiment needs at least one run.")
    if engine == "pool":
        _split(cfg, str(exp.pool["split"]), "pool")
    return exp
