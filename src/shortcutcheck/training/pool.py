"""RQ3 model pool (roadmap step 3.5), exactly as PREREGISTRATION.md section 5 and D18 fix it.

Full-feature models only, trained on the 2012-14 training partition of the RQ3 split:
5 algorithms x 5 seeds x (6 settings at skew 0 + 2 settings at each of 3 injected levels)
= 300 models in 60 configurations. Behaviour-only models are never built here: they score 0
on MR1-MR4 by construction and would inflate the RQ3 correlation for free.

Skew injection on ``apk_size`` (``inject_skew``): "contrary" apps are malware below the
training median or goodware at or above it. Level q removes ceil(q x contrary) of each class's
contrary apps at random (seed = model seed). Every level is then cut at random, per class, to
the sizes left at the highest level, so the amount of data is the same at every level.

This step opens no test data. Each model is saved (``data/models/rq3/``, git-ignored) for the
signal (3.6) and ground-truth (3.7) steps, and described in ``pool_models.csv`` with a hash of
its validation probabilities, so a rebuilt pool can be checked for identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.experiments.spec import Experiment
from shortcutcheck.runner.engine import single_threaded
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select


@dataclass(frozen=True)
class SkewPlan:
    """Per-class sizes every skew level is cut to, and the contrary rule's median."""

    column: str
    median: float
    max_level: float
    target: dict[int, int]  # class -> training apps kept at every level


def contrary_mask(x: pd.DataFrame, y: pd.Series, column: str, median: float) -> np.ndarray:
    v = x[column].to_numpy(dtype=float)
    lab = y.to_numpy()
    return np.asarray(((lab == 1) & (v < median)) | ((lab == 0) & (v >= median)), dtype=bool)


def skew_plan(x: pd.DataFrame, y: pd.Series, column: str, levels: list[float]) -> SkewPlan:
    median = float(np.nanmedian(x[column].to_numpy(dtype=float)))
    contrary = contrary_mask(x, y, column, median)
    lab = y.to_numpy()
    q = max(levels)
    target = {
        int(c): int((lab == c).sum() - math.ceil(q * (contrary & (lab == c)).sum()))
        for c in sorted(set(lab.tolist()))
    }
    return SkewPlan(column, median, q, target)


def inject_skew(
    x: pd.DataFrame, y: pd.Series, plan: SkewPlan, level: float, seed: int
) -> np.ndarray:
    """Row positions of the training set for one skew level and seed (sorted)."""
    rng = np.random.default_rng(seed)
    contrary = contrary_mask(x, y, plan.column, plan.median)
    lab = y.to_numpy()
    keep = []
    for c, size in plan.target.items():
        cls = lab == c
        bad = np.flatnonzero(cls & contrary)
        good = np.flatnonzero(cls & ~contrary)
        n_remove = math.ceil(level * len(bad))
        removed = rng.choice(bad, size=n_remove, replace=False) if n_remove else np.array([], int)
        kept = np.setdiff1d(np.concatenate([bad, good]), removed)
        keep.append(np.sort(rng.choice(kept, size=size, replace=False)))
    return np.sort(np.concatenate(keep))


@dataclass(frozen=True)
class PoolMember:
    model: str
    algorithm: str
    skew: float
    setting: int
    seed: int
    params: dict[str, Any]

    @property
    def configuration(self) -> str:
        return f"{self.algorithm}|{self.skew:g}|{self.setting}"


def pool_members(exp: Experiment) -> list[PoolMember]:
    """Every model of the pool, in a fixed order."""
    pool = exp.pool
    if tuple(exp.feature_sets) != ("full",):
        raise ValueError("The RQ3 pool holds full-feature models only (behaviour-only excluded).")
    out = []
    for algorithm in exp.algorithms:
        settings = pool["hyperparameters"][algorithm]
        for level, n_settings in zip(
            pool["skew"]["levels"], pool["skew"]["settings_per_level"], strict=True
        ):
            for setting in range(n_settings):
                for seed in exp.seeds:
                    name = f"{algorithm}-q{float(level):g}-h{setting}-s{seed}"
                    out.append(
                        PoolMember(
                            name, algorithm, float(level), setting, seed, dict(settings[setting])
                        )
                    )
    return out


def probability_hash(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype=np.float64).tobytes()).hexdigest()[:16]


def build_pool(
    cfg: Config,
    exp: Experiment,
    commit: str,
    model_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Train and save every pool model; return one row per model. No test data is read."""
    if exp.engine != "pool":
        raise ValueError(f"'{exp.experiment}' is not a pool experiment.")
    split = str(exp.pool["split"])
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x, y = select(load_part(cfg, split, "train"), "full", *args)
    x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
    x_val = x_val.reindex(columns=x.columns)
    skew = exp.pool["skew"]
    plan = skew_plan(x, y, str(skew["column"]), [float(v) for v in skew["levels"]])
    contrary = contrary_mask(x, y, plan.column, plan.median)
    model_dir.mkdir(parents=True, exist_ok=True)
    _setup_mlflow(cfg, exp.mlflow_experiment)
    rows = []
    for m in pool_members(exp):
        if progress:
            progress(m.model)
        rows_idx = inject_skew(x, y, plan, m.skew, m.seed)
        xt, yt = x.iloc[rows_idx], y.iloc[rows_idx]
        model = build_model(m.algorithm, m.seed, params=m.params).fit(xt, yt)  # type: ignore[arg-type]
        with single_threaded(model):  # parallel tree averaging is not bit-reproducible (D14)
            p = model.predict_proba(x_val)[:, 1]
        pred = (p >= 0.5).astype(int)
        path = model_dir / f"{m.model}.joblib"
        joblib.dump(model, path, compress=3)
        with mlflow.start_run(run_name=m.model) as active:
            mlflow.set_tags(
                {
                    "step": "3.5",
                    "experiment_config": exp.path.as_posix(),
                    "split": split,
                    "configuration": m.configuration,
                    "git_commit": commit,
                }
            )
            mlflow.log_params(
                {
                    "algorithm": m.algorithm,
                    "skew": m.skew,
                    "setting": m.setting,
                    "seed": m.seed,
                    "params": json.dumps(m.params, sort_keys=True),
                    "n_train": len(rows_idx),
                }
            )
            val_f1 = float(f1_score(y_val, pred))
            mlflow.log_metrics({"val_f1": val_f1})
            run_id = active.info.run_id
        rows.append(
            {
                "model": m.model,
                "algorithm": m.algorithm,
                "skew": m.skew,
                "setting": m.setting,
                "seed": m.seed,
                "configuration": m.configuration,
                "params": json.dumps(m.params, sort_keys=True),
                "feature_set": "full",
                "n_features": len(model.feature_names_in_),
                "n_goodware": int((yt == 0).sum()),
                "n_malware": int((yt == 1).sum()),
                "contrary_share": float(contrary[rows_idx].mean()),
                "val_f1": val_f1,
                "val_accuracy": float(accuracy_score(y_val, pred)),
                "val_probability_hash": probability_hash(p),
                "git_commit": commit,
                "mlflow_run": run_id,
            }
        )
    return pd.DataFrame(rows)


def render_pool(pool: pd.DataFrame, plan_median: float, commit: str) -> str:
    by = (
        pool.groupby(["skew"], sort=True)
        .agg(
            models=("model", "count"),
            configurations=("configuration", "nunique"),
            goodware=("n_goodware", "first"),
            malware=("n_malware", "first"),
            contrary=("contrary_share", "mean"),
            val_f1=("val_f1", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# RQ3 model pool (roadmap step 3.5)",
        "",
        f"Generated by `shortcutcheck experiment run configs/experiments/rq3.yaml` at commit "
        f"`{commit}`. Training partition of `rq3-AndroCT-2012-14-vs-2017-19` (2012-14 apps); "
        "validation scores use 2012-14 validation labels. No test data was opened. Per-model "
        "details and MLflow run IDs: `pool_models.csv`.",
        "",
        f"{len(pool)} models in {pool['configuration'].nunique()} configurations; feature sets: "
        f"{', '.join(sorted(set(pool['feature_set'])))} (behaviour-only models excluded). "
        f"Skew median of `apk_size`: {plan_median:,.1f} bytes.",
        "",
        "| Skew level | Models | Configurations | Goodware | Malware | Contrary share "
        "| Mean validation F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in by.itertuples():
        lines.append(
            f"| {r.skew:g} | {r.models} | {r.configurations} | {r.goodware} | {r.malware} "
            f"| {r.contrary:.3f} | {r.val_f1:.3f} |"
        )
    algo = (
        pool.groupby(["algorithm", "skew"], sort=False)["val_f1"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    lines += [
        "",
        "| Algorithm | Skew | Mean validation F1 | Range |",
        "| --- | ---: | ---: | --- |",
    ]
    for t in algo.itertuples():
        lines.append(f"| {t.algorithm} | {t.skew:g} | {t.mean:.3f} | {t.min:.3f}–{t.max:.3f} |")
    return "\n".join(lines) + "\n"
