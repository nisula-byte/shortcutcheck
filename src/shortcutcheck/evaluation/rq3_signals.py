"""Roadmap step 3.6: label-free signals for every RQ3 pool model (PREREGISTRATION.md section 5).

The 2017-19 test partition is opened ONCE here, through ``load_test()``, for its inputs only:
the label column is dropped in ``unlabelled_test_inputs`` before anything else sees the frame,
and every signal function rejects a frame that still carries one. Test labels are used only in
step 3.7 (F1 drop), with one logged ``load_test()`` call per model.

Per model (all predictions single-threaded):

* pre-deployment, on the 2012-14 validation partition (no labels): MR1/MR3/MR4 violation
  rates with the realism filter (``mr4_pre`` is the primary signal), the unfiltered MR4 rate,
  the core-MR mean, and the SHAP metadata share (500 validation inputs, 100 training apps as
  background, both uniform samples with seed 11);
* monitoring, on 2017-19 inputs (labels unused): MR4 rate on a uniform 5,000-app sample
  (seed 11), mean confidence drop, PSI and the ATC predicted accuracy drop (ATC threshold from
  validation labels only).

MRs and the realism check are fitted on the RQ3 training partition (the input population).
Models run in parallel worker processes; every value depends only on the model and fixed seeds,
so the result does not depend on the number of workers.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from shortcutcheck import signals as sig
from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.experiment import fit_mrs
from shortcutcheck.evaluation.locked_test import load_test
from shortcutcheck.experiments.spec import Experiment
from shortcutcheck.mrs.base import MetamorphicRelation
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.runner.engine import run_mr, single_threaded
from shortcutcheck.training.features import select
from shortcutcheck.training.pool import PoolMember, pool_members

RECHECK_PER_ALGORITHM = 2

SIGNAL_COLUMNS = (
    "mr4_pre",
    "mr1_pre",
    "mr3_pre",
    "core_mean_pre",
    "mr4_pre_unfiltered",
    "shap_share",
    "mr4_monitor",
    "confidence_drop",
    "psi",
    "atc_drop",
)


@dataclass
class Context:
    """Everything a worker needs; built once and sent to each worker process."""

    x_val: pd.DataFrame
    y_val: pd.Series  # VALIDATION labels only: ATC threshold and validation accuracy
    x_test: pd.DataFrame  # 2017-19 inputs, no label column
    test_sample: np.ndarray
    x_explain: pd.DataFrame
    x_background: pd.DataFrame
    mrs: list[MetamorphicRelation]
    realism: RealismCheck
    metadata: list[str]
    model_dir: Path
    psi_bins: int
    psi_floor: float


def unlabelled_test_inputs(cfg: Config, split: str, purpose: str) -> pd.DataFrame:
    """The test partition's feature columns only. The labels never leave this function."""
    frame = load_test(cfg, split, purpose=purpose)
    x, _labels = select(frame, "full", cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    del _labels, frame
    return sig.assert_unlabelled(x)


def build_context(
    cfg: Config, exp: Experiment, model_dir: Path, x_test: pd.DataFrame | None = None
) -> Context:
    split = str(exp.pool["split"])
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_train, y_train = select(load_part(cfg, split, "train"), "full", *args)
    x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
    x_val = x_val.reindex(columns=x_train.columns)
    if x_test is None:
        purpose = (
            f"rq3 signals (roadmap step 3.6), config {exp.path.as_posix()}: 2017-19 INPUTS ONLY "
            "for the monitoring signals; the label column is dropped before use"
        )
        x_test = unlabelled_test_inputs(cfg, split, purpose)
    x_test = sig.assert_unlabelled(x_test).reindex(columns=x_train.columns)
    signals = exp.pool["signals"]
    shap_cfg = signals["pre_deployment"]["shap_metadata_share"]
    rng = np.random.default_rng(exp.sample_seed)
    explain = np.sort(
        rng.choice(len(x_val), size=min(int(shap_cfg["inputs"]), len(x_val)), replace=False)
    )
    rng = np.random.default_rng(exp.sample_seed)
    background = np.sort(
        rng.choice(len(x_train), size=min(int(shap_cfg["background"]), len(x_train)), replace=False)
    )
    n_test = len(x_test)
    size = n_test if exp.mr_sample is None else min(exp.mr_sample, n_test)
    rng = np.random.default_rng(exp.sample_seed)
    test_sample = np.sort(rng.choice(n_test, size=size, replace=False))
    meta = [c for c in cfg.metadata_columns if c in x_train.columns]
    psi_cfg = signals["monitoring"]["psi"]
    return Context(
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        test_sample=test_sample,
        x_explain=x_val.iloc[explain].reset_index(drop=True),
        x_background=x_train.iloc[background].reset_index(drop=True),
        mrs=fit_mrs(cfg, x_train, y_train, ("MR1", "MR3", "MR4")),
        realism=RealismCheck(seed=exp.seeds[0], metadata_columns=meta).fit(x_train),
        metadata=meta,
        model_dir=model_dir,
        psi_bins=int(psi_cfg["bins"]),
        psi_floor=float(psi_cfg["epsilon"]),
    )


def model_signals(ctx: Context, member: PoolMember) -> dict[str, Any]:
    """Every pre-registered signal for one pool model."""
    sig.assert_unlabelled(ctx.x_test)
    model = joblib.load(ctx.model_dir / f"{member.model}.joblib")
    cols = list(model.feature_names_in_)
    with single_threaded(model):
        p_val = model.predict_proba(ctx.x_val[cols])[:, 1]
        p_test = model.predict_proba(ctx.x_test[cols])[:, 1]
    nan = float("nan")
    out: dict[str, Any] = {
        "model": member.model,
        "mr1_pre": nan,
        "mr3_pre": nan,
        "mr4_pre": nan,
        "mr4_pre_unfiltered": nan,
        "mr4_monitor": nan,
    }
    for mr in ctx.mrs:  # an MR that does not apply to the split leaves its signals NaN
        res = run_mr(model, mr, ctx.x_val, seed=member.seed, realism=ctx.realism)
        out[f"{mr.mr_id.lower()}_pre"] = res.violation_rate
        if mr.mr_id == "MR4":
            out["mr4_pre_unfiltered"] = res.violation_rate_unfiltered
            xs = ctx.x_test.iloc[ctx.test_sample].reset_index(drop=True)
            out["mr4_monitor"] = run_mr(
                model, mr, xs, seed=member.seed, realism=ctx.realism
            ).violation_rate
    out["core_mean_pre"] = float(np.mean([out["mr1_pre"], out["mr3_pre"], out["mr4_pre"]]))
    # (NaN if any core MR does not apply: the mean is then not the pre-registered quantity.)
    values = sig.shap_values(model, member.algorithm, ctx.x_explain, ctx.x_background, seed=11)
    out["shap_share"] = sig.metadata_share(values, cols, ctx.metadata)
    correct = (p_val >= 0.5).astype(int) == ctx.y_val.to_numpy()
    val_accuracy = float(correct.mean())
    threshold = sig.atc_threshold(sig.confidence(p_val), correct)
    out["confidence_drop"] = sig.confidence_drop(p_val, p_test)
    out["psi"] = sig.psi(p_val, p_test, ctx.psi_bins, ctx.psi_floor)
    out["atc_drop"] = sig.atc_drop(val_accuracy, threshold, sig.confidence(p_test))
    out["atc_threshold"] = threshold
    out["val_accuracy"] = val_accuracy
    out["mean_p_test"] = float(p_test.mean())
    return out


_CTX: Context | None = None


def _init(ctx: Context) -> None:
    global _CTX
    _CTX = ctx
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _work(member: PoolMember) -> dict[str, Any]:
    assert _CTX is not None
    return model_signals(_CTX, member)


# Slowest first, so two workers finish together (SHAP: random forest > MLP > SVM > others).
COST = {"rf": 0, "mlp": 1, "svm": 2, "lgbm": 3, "lr": 4}


def compute_signals(
    exp: Experiment,
    ctx: Context,
    *,
    workers: int = 2,
    only: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Signals for every pool model (or the models in ``only``), in pool order."""
    members = [m for m in pool_members(exp) if only is None or m.model in only]
    missing = [m.model for m in members if not (ctx.model_dir / f"{m.model}.joblib").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pool models missing (run step 3.5): {missing[:3]}")
    order = sorted(members, key=lambda m: (COST[m.algorithm], m.model))
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        _init(ctx)
        for m in order:
            rows.append(_work(m))
            if progress:
                progress(m.model)
    else:
        # "spawn", not "fork": forking after LightGBM/MLflow threads start can deadlock.
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=spawn, initializer=_init, initargs=(ctx,)
        ) as ex:
            for m, row in zip(order, ex.map(_work, order), strict=True):
                rows.append(row)
                if progress:
                    progress(m.model)
    by_model = {r["model"]: r for r in rows}
    table = pd.DataFrame(
        [
            {
                "model": m.model,
                "algorithm": m.algorithm,
                "skew": m.skew,
                "setting": m.setting,
                "seed": m.seed,
                "configuration": m.configuration,
                **{k: v for k, v in by_model[m.model].items() if k != "model"},
            }
            for m in members
        ]
    )
    return table


def render_signals(table: pd.DataFrame, commit: str) -> str:
    """Distribution of every signal by algorithm and skew level. Contains no test labels."""
    lines = [
        "# RQ3 label-free signals (roadmap step 3.6)",
        "",
        f"Generated by `shortcutcheck experiment signals configs/experiments/rq3.yaml` at commit "
        f"`{commit}`. One row per pool model in `signals.csv`. The 2017-19 partition was opened "
        "once for its inputs; its labels were dropped before any signal was computed and are "
        "not used here. Definitions: PREREGISTRATION.md section 5.",
        "",
        f"{len(table)} models. Missing values per signal: "
        + ", ".join(f"{c} {int(table[c].isna().sum())}" for c in SIGNAL_COLUMNS)
        + ".",
        "",
        "| Algorithm | Skew | " + " | ".join(SIGNAL_COLUMNS) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(SIGNAL_COLUMNS)) + " |",
    ]
    for (a, s), g in table.groupby(["algorithm", "skew"], sort=False):
        cells = [f"{g[c].mean():.3f}" for c in SIGNAL_COLUMNS]
        lines.append(f"| {a} | {s:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
