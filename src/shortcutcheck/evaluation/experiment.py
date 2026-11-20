"""The ``models`` engine: train models on frozen splits and run MRs on evaluation inputs.

Used by the pilot (step 2.7), RQ1 (3.3) and RQ2 (3.4). It generalises the pilot loop and, for
the pilot config, produces exactly the pilot's numbers (``make pilot-compare`` checks this).

Rules this module enforces:

* Test data is read only through ``load_test``, at most once per split per experiment, with
  the purpose logged. Labels are used only for the detection-quality table; MRs never see them.
* No model is scored on an app it was trained on: apps in the model's training split (and in
  any ``exclude_train_of`` split) are removed from its evaluation inputs.
* MRs and the realism check are fitted on the training partition of the evaluation split.
* When an evaluation partition has more than ``mr_sample`` apps, MRs run on one uniform random
  sample (``sample_seed``) drawn without labels and shared by every model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part, read_index
from shortcutcheck.evaluation.locked_test import load_test
from shortcutcheck.experiments.spec import Experiment, ModelRun, Target
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError
from shortcutcheck.mrs.core import core_mrs
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.mrs.secondary import secondary_mrs
from shortcutcheck.runner.engine import run_mr
from shortcutcheck.runner.stats import MRSummary, summarise
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select

# Columns that identify a row; the pilot CSV keeps only the last four of the first block.
KEY_COLUMNS = ["run", "train_split", "arm", "eval_split", "partition", "role"]


def lower_bound(s: MRSummary) -> float:
    """The CI lower bound the reports show: Wilson at 0 or all violations, else bootstrap."""
    if s.n_counted == 0:
        return float("nan")
    return s.rate_wilson.low if s.n_violations in (0, s.n_counted) else s.rate_ci.low


def upper_bound(s: MRSummary) -> float:
    if s.n_counted == 0:
        return float("nan")
    return s.rate_wilson.high if s.n_violations in (0, s.n_counted) else s.rate_ci.high


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    commit: str
    rows: pd.DataFrame  # one row per (run, model, target, MR, stratum)
    detection: pd.DataFrame  # one row per (run, model, target)
    inputs: pd.DataFrame  # one row per (run, target): how many inputs and exclusions


@dataclass
class _Prepared:
    """One evaluation target, ready to use: inputs, labels, fitted MRs and realism check."""

    x: pd.DataFrame
    y: pd.Series
    sample: np.ndarray  # row positions MRs run on
    valid_year: np.ndarray
    mrs: list[MetamorphicRelation]
    realism: RealismCheck | None
    n_before_exclusion: int


def fit_mrs(
    cfg: Config, x_train: pd.DataFrame, y_train: pd.Series, wanted: tuple[str, ...]
) -> list[MetamorphicRelation]:
    """Every wanted MR that applies, fitted on training data. Inapplicable ones are skipped."""
    valid = (cfg.android_first_year, cfg.collection_year)
    fitted = []
    for mr in [*core_mrs(list(x_train.columns), valid), *secondary_mrs(list(x_train.columns))]:
        if mr.mr_id not in wanted:
            continue
        try:
            fitted.append(mr.fit(x_train, y_train))
        except NotApplicableError:
            continue
    return fitted


def stratified_subsample(y: pd.Series, n: int, seed: int) -> np.ndarray:
    """Row positions of a size-``n`` sample that keeps the class shares of ``y`` (rounded)."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    values = y.to_numpy()
    picks = []
    classes = sorted(set(values.tolist()))
    remaining = n
    for i, c in enumerate(classes):
        idx = np.flatnonzero(values == c)
        k = remaining if i == len(classes) - 1 else round(n * len(idx) / len(values))
        k = min(k, len(idx))
        remaining -= k
        picks.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(picks))


class _Inputs:
    """Loads evaluation partitions, opening each test partition at most once."""

    def __init__(self, cfg: Config, exp: Experiment) -> None:
        self.cfg = cfg
        self.exp = exp
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    def frame(self, split: str, partition: str) -> pd.DataFrame:
        key = (split, partition)
        if key not in self._frames:
            if partition == "test":
                purpose = (
                    f"{self.exp.experiment} (roadmap step {self.exp.step}), config "
                    f"{self.exp.path.as_posix()}: MR inputs (labels unused) and the detection "
                    "table (labels used once)"
                )
                self._frames[key] = load_test(self.cfg, split, purpose=purpose)
            else:
                self._frames[key] = load_part(self.cfg, split, partition)
        return self._frames[key]


def _prepare(
    cfg: Config,
    exp: Experiment,
    run: ModelRun,
    target: Target,
    inputs: _Inputs,
    columns: pd.Index,
) -> _Prepared:
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    frame = inputs.frame(target.split, target.partition)
    excluded: set[str] = set()
    for split in {run.train, *target.exclude_train_of}:
        excluded |= read_index(cfg, split, "train")
    keep = ~frame[cfg.id_column].isin(excluded)
    x, y = select(frame[keep].reset_index(drop=True), "full", *args)
    x = x.reindex(columns=columns)

    x_fit, y_fit = select(load_part(cfg, target.split, "train"), "full", *args)
    x_fit = x_fit.reindex(columns=columns)
    meta = [c for c in cfg.metadata_columns if c in columns]
    realism = (
        RealismCheck(seed=exp.seeds[0], metadata_columns=meta).fit(x_fit) if exp.realism else None
    )
    n = len(x)
    if exp.mr_sample is not None and n > exp.mr_sample:
        rng = np.random.default_rng(exp.sample_seed)
        sample = np.sort(rng.choice(n, size=exp.mr_sample, replace=False))
    else:
        sample = np.arange(n)
    year = x["year"].to_numpy(dtype=float) if "year" in x.columns else np.full(n, np.nan)
    valid_year = (year >= cfg.android_first_year) & (year <= cfg.collection_year)
    return _Prepared(
        x=x,
        y=y,
        sample=sample,
        valid_year=valid_year,
        mrs=fit_mrs(cfg, x_fit, y_fit, exp.mrs),
        realism=realism,
        n_before_exclusion=len(frame),
    )


def _row(
    base: dict[str, object], stratum: str, s: MRSummary, max_delta: float
) -> dict[str, object]:
    return {
        **base,
        "stratum": stratum,
        "n_counted": s.n_counted,
        "n_violations": s.n_violations,
        "n_unrealistic": s.n_unrealistic,
        "rate": s.rate,
        "rate_low": lower_bound(s),
        "rate_high": upper_bound(s),
        "rate_unfiltered": s.rate_unfiltered,
        "delta": s.delta,
        "delta_low": s.delta_ci.low,
        "delta_high": s.delta_ci.high,
        "max_abs_delta": max_delta,
    }


def _training_data(
    cfg: Config, run: ModelRun, seed: int, x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """The model's training data: the whole partition, or a size-matched sample per seed."""
    if run.arm == "size-matched" and run.size_like is not None:
        n = len(read_index(cfg, run.size_like, "train"))
        keep = stratified_subsample(y, n, seed)
        x, y = x.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    return x, y


def _evaluate(
    model: Any,
    p: _Prepared,
    cols: list[str],
    key: dict[str, object],
    prefix: str,
    seed: int,
    mlflow_run: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Detection quality on all inputs, then every MR on the MR sample, for one target."""
    scores = model.predict_proba(p.x[cols])[:, 1]
    quality = detection_metrics(p.y, scores)
    mlflow.log_metrics({prefix + k: v for k, v in quality.items()})
    xs = p.x.iloc[p.sample].reset_index(drop=True)
    valid = p.valid_year[p.sample]
    rows: list[dict[str, object]] = []
    for mr in p.mrs:
        res = run_mr(model, mr, xs, seed=seed, realism=p.realism)
        delta = np.abs(res.p_after - res.p_before)[res.counted]
        max_delta = float(delta.max()) if delta.size else float("nan")
        info = {**key, "mr": mr.mr_id, "mlflow_run": mlflow_run}
        for stratum, mask in (("all", None), ("valid year", valid), ("invalid year", ~valid)):
            rows.append(_row(info, stratum, summarise(res, mask=mask, seed=seed), max_delta))
        overall = summarise(res, seed=seed)
        tag = f"{prefix}{mr.mr_id}"
        mlflow.log_metrics(
            {
                f"{tag}_rate": overall.rate,
                f"{tag}_rate_low": lower_bound(overall),
                f"{tag}_n_counted": float(overall.n_counted),
                f"{tag}_rate_unfiltered": overall.rate_unfiltered,
                f"{tag}_delta": overall.delta,
            }
        )
    return rows, {**key, **quality, "n_inputs": len(p.x)}


def run_experiment(
    cfg: Config,
    exp: Experiment,
    commit: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Run every model of a ``models`` experiment. Deterministic for a fixed config."""
    if exp.engine != "models":
        raise ValueError(f"'{exp.experiment}' uses the '{exp.engine}' engine, not 'models'.")
    _setup_mlflow(cfg, exp.mlflow_experiment)
    inputs = _Inputs(cfg, exp)
    rows: list[dict[str, object]] = []
    detection: list[dict[str, object]] = []
    sizes: list[dict[str, object]] = []
    for run in exp.runs:
        x_all, y_all = select(
            load_part(cfg, run.train, "train"),
            "full",
            cfg.metadata_columns,
            cfg.id_column,
            cfg.dropped_features,
        )
        columns = x_all.columns
        behaviour = [c for c in columns if c not in cfg.metadata_columns]
        prepared = {t.role: _prepare(cfg, exp, run, t, inputs, columns) for t in run.targets}
        for t in run.targets:
            p = prepared[t.role]
            sizes.append(
                {
                    "run": run.name,
                    "eval_split": t.split,
                    "partition": t.partition,
                    "role": t.role,
                    "n_partition": p.n_before_exclusion,
                    "n_excluded": p.n_before_exclusion - len(p.x),
                    "n_inputs": len(p.x),
                    "n_mr_sample": len(p.sample),
                    "mrs": " ".join(m.mr_id for m in p.mrs),
                }
            )
        for algorithm in exp.algorithms:
            for feature_set in exp.feature_sets_for(run):
                cols = list(columns) if feature_set == "full" else behaviour
                for seed in exp.seeds:
                    if progress:
                        progress(f"{run.name} {algorithm} {feature_set} s{seed}")
                    x_train, y_train = _training_data(cfg, run, seed, x_all, y_all)
                    model = build_model(algorithm, seed).fit(x_train[cols], y_train)  # type: ignore[arg-type]
                    base = {"algorithm": algorithm, "feature_set": feature_set, "seed": seed}
                    name = f"{run.name}-{algorithm}-{feature_set}-s{seed}"
                    with mlflow.start_run(run_name=name) as active:
                        mlflow.set_tags(
                            {
                                "step": exp.step,
                                "experiment_config": exp.path.as_posix(),
                                "run": run.name,
                                "split": run.train,
                                "arm": run.arm,
                                "git_commit": commit,
                            }
                        )
                        mlflow.log_params({**base, "n_train": len(x_train)})
                        for t in run.targets:
                            key = {
                                "run": run.name,
                                "train_split": run.train,
                                "arm": run.arm,
                                "eval_split": t.split,
                                "partition": t.partition,
                                "role": t.role,
                            }
                            prefix = "" if len(run.targets) == 1 else f"{t.role}_"
                            r, d = _evaluate(
                                model,
                                prepared[t.role],
                                cols,
                                {**key, **base},
                                prefix,
                                seed,
                                active.info.run_id,
                            )
                            rows += r
                            detection.append(d)
    return ExperimentResult(
        exp.experiment, commit, pd.DataFrame(rows), pd.DataFrame(detection), pd.DataFrame(sizes)
    )
