"""Addendum D runs: H4 on the RQ1/RQ2 test partitions, H5 on the 2015-16 holdout.

Data access (PREREGISTRATION-ADDENDUM-D.md): every test partition and the holdout are opened
once per run through ``load_test()`` with the purpose logged. Labels are separated from the
inputs straight away; MRs and label-free signals see inputs only. With ``dry_run=True`` the
validation partition of the same split replaces every test partition and the holdout, nothing
is logged as a test access, and the output is not a result.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from shortcutcheck import signals as sig
from shortcutcheck.addenda import addendum_d as rules
from shortcutcheck.addenda.fix_poc import augment, contrary_apps
from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.experiment import fit_mrs
from shortcutcheck.evaluation.locked_test import load_test
from shortcutcheck.evaluation.rq3_signals import build_context
from shortcutcheck.experiments.spec import Experiment
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.mrs.base import MetamorphicRelation
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.runner.engine import run_mr, single_threaded
from shortcutcheck.runner.stats import summarise
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select
from shortcutcheck.training.pool import PoolMember, pool_members, probability_hash

Progress = Callable[[str], None]


def _quiet(_m: str) -> None:
    return None


# ---- shared pieces ------------------------------------------------------------------------


@dataclass
class TrainData:
    x: pd.DataFrame
    y: pd.Series
    mrs: dict[str, MetamorphicRelation]
    realism: RealismCheck


@dataclass
class EvalData:
    x: pd.DataFrame
    y: pd.Series
    sample: np.ndarray
    contrary: np.ndarray


def train_data(cfg: Config, split: str) -> TrainData:
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x, y = select(load_part(cfg, split, "train"), "full", *args)
    meta = [c for c in cfg.metadata_columns if c in x.columns]
    realism = RealismCheck(seed=rules.SAMPLE_SEED, metadata_columns=meta).fit(x)
    mrs = {m.mr_id: m for m in fit_mrs(cfg, x, y, rules.EVAL_MRS)}
    return TrainData(x, y, mrs, realism)


def open_frame(cfg: Config, split: str, purpose: str, dry_run: bool) -> pd.DataFrame:
    """The test partition (logged), or in a dry run the same split's validation partition."""
    if dry_run:
        return load_part(cfg, split, "val")
    return load_test(cfg, split, purpose=purpose)


def eval_data(cfg: Config, frame: pd.DataFrame, tr: TrainData) -> EvalData:
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x, y = select(frame, "full", *args)
    x = sig.assert_unlabelled(x.reindex(columns=tr.x.columns))
    n = len(x)
    if n > rules.MR_SAMPLE:
        rng = np.random.default_rng(rules.SAMPLE_SEED)
        sample = np.sort(rng.choice(n, size=rules.MR_SAMPLE, replace=False))
    else:
        sample = np.arange(n)
    return EvalData(x, y, sample, contrary_apps(tr.x, tr.y, x, y))


def measure(
    model: Any, ev: EvalData, tr: TrainData, seed: int, mrs: Sequence[str] = rules.EVAL_MRS
) -> dict[str, float]:
    """F1, ROC-AUC, contrary-app accuracy on all inputs; MR rates on the MR sample."""
    with single_threaded(model):
        scores = model.predict_proba(ev.x)[:, 1]
        q = detection_metrics(ev.y, scores)
        correct = (scores >= 0.5).astype(int) == ev.y.to_numpy().astype(int)
        out: dict[str, float] = {
            "f1": q["f1"],
            "roc_auc": q["roc_auc"],
            "contrary_acc": float(correct[ev.contrary].mean()) if ev.contrary.any() else np.nan,
            "n_contrary": float(ev.contrary.sum()),
            "n_eval": float(len(ev.y)),
        }
        xs = ev.x.iloc[ev.sample].reset_index(drop=True)
        for mr_id in mrs:
            if mr_id not in tr.mrs:
                out[f"rate_{mr_id}"], out[f"n_{mr_id}"] = np.nan, 0.0
                continue
            s = summarise(run_mr(model, tr.mrs[mr_id], xs, seed=seed, realism=tr.realism))
            out[f"rate_{mr_id}"], out[f"n_{mr_id}"] = s.rate, float(s.n_counted)
    return out


def arm_data(tr: TrainData, arm: str, seed: int) -> tuple[pd.DataFrame, pd.Series, int]:
    if arm == "baseline":
        return tr.x, tr.y, 0
    if arm == "augmented":
        return augment(tr.x, tr.y, tr.mrs["MR4"], tr.realism, seed)
    if arm == "duplicate":
        return (
            pd.concat([tr.x, tr.x], ignore_index=True),
            pd.concat([tr.y, tr.y], ignore_index=True),
            0,
        )
    raise ValueError(f"unknown arm {arm!r}")


def _log(
    log: bool, name: str, commit: str, params: dict[str, Any], metrics: dict[str, float]
) -> str:
    if not log:
        return ""
    with mlflow.start_run(run_name=name) as r:
        mlflow.set_tags({"git_commit": commit, "addendum": "D"})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not pd.isna(v)})
        return str(r.info.run_id)


def _purpose(what: str) -> str:
    return f"addendum D (PREREGISTRATION-ADDENDUM-D.md): {what}"


# ---- H4 -----------------------------------------------------------------------------------


def run_h4(
    cfg: Config,
    commit: str,
    *,
    dry_run: bool = False,
    splits: Sequence[str] = rules.H4_SPLITS,
    algorithms: Sequence[str] = rules.ALGORITHMS,
    seeds: Sequence[int] = rules.SEEDS,
    progress: Progress = _quiet,
) -> pd.DataFrame:
    log = not dry_run
    if log:
        _setup_mlflow(cfg, rules.MLFLOW_EXPERIMENT)
    rows = []
    for split in splits:
        tr = train_data(cfg, split)
        frame = open_frame(
            cfg,
            split,
            _purpose(
                f"H4, models trained on {split} train; MR inputs (labels unused), F1 and "
                "contrary-app accuracy (labels used once)"
            ),
            dry_run,
        )
        ev = eval_data(cfg, frame, tr)
        del frame
        arms = [a for a in rules.ARMS if not (a == "duplicate" and split in rules.NO_CONTROL)]
        for seed in seeds:
            data = {arm: arm_data(tr, arm, seed) for arm in arms}
            for algorithm in algorithms:
                for arm in arms:
                    progress(f"H4 {split} {algorithm} s{seed} {arm}")
                    x, y, rejected = data[arm]
                    model = build_model(algorithm, seed).fit(x, y)  # type: ignore[arg-type]
                    m = measure(model, ev, tr, seed)
                    row: dict[str, Any] = {
                        "split": split,
                        "algorithm": algorithm,
                        "seed": seed,
                        "arm": arm,
                        "n_train": len(y),
                        "n_rejected_copies": rejected,
                        **m,
                    }
                    row["mlflow_run"] = _log(
                        log,
                        f"H4-{split}-{algorithm}-s{seed}-{arm}",
                        commit,
                        {k: row[k] for k in ("split", "algorithm", "seed", "arm", "n_train")},
                        m,
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


# ---- H5 -----------------------------------------------------------------------------------


def verify_pool(cfg: Config, exp: Experiment, model_dir: Path, members: list[PoolMember]) -> None:
    """Stop unless every pool model reproduces its step 3.5 validation probability hash."""
    recorded = pd.read_csv(cfg.reports_dir / "rq3" / "pool_models.csv").set_index("model")
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_val, _ = select(load_part(cfg, str(exp.pool["split"]), "val"), "full", *args)
    bad = []
    for m in members:
        model = joblib.load(model_dir / f"{m.model}.joblib")
        with single_threaded(model):
            p = model.predict_proba(x_val[list(model.feature_names_in_)])[:, 1]
        if probability_hash(p) != recorded.loc[m.model, "val_probability_hash"]:
            bad.append(m.model)
    if bad:
        raise RuntimeError(f"{len(bad)} pool models differ from step 3.5: {bad[:5]}")


_CTX: Any = None


def _init(ctx: Any) -> None:
    global _CTX
    _CTX = ctx
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def monitoring_signals(ctx: Any, member: PoolMember) -> dict[str, Any]:
    """Label-free monitoring signals of one pool model on the holdout inputs (as in step 3.6)."""
    sig.assert_unlabelled(ctx.x_test)
    model = joblib.load(ctx.model_dir / f"{member.model}.joblib")
    cols = list(model.feature_names_in_)
    with single_threaded(model):
        p_val = model.predict_proba(ctx.x_val[cols])[:, 1]
        p_new = model.predict_proba(ctx.x_test[cols])[:, 1]
        mr4 = next(m for m in ctx.mrs if m.mr_id == "MR4")
        xs = ctx.x_test.iloc[ctx.test_sample].reset_index(drop=True)
        rate = run_mr(model, mr4, xs, seed=member.seed, realism=ctx.realism).violation_rate
    correct = (p_val >= 0.5).astype(int) == ctx.y_val.to_numpy()
    threshold = sig.atc_threshold(sig.confidence(p_val), correct)
    return {
        "model": member.model,
        "mr4_monitor": rate,
        "confidence_drop": sig.confidence_drop(p_val, p_new),
        "psi": sig.psi(p_val, p_new, ctx.psi_bins, ctx.psi_floor),
        "atc_drop": sig.atc_drop(float(correct.mean()), threshold, sig.confidence(p_new)),
        "p_holdout": p_new,
        "p_val": p_val,
    }


def _work(member: PoolMember) -> dict[str, Any]:
    return monitoring_signals(_CTX, member)


def run_h5(
    cfg: Config,
    exp: Experiment,
    commit: str,
    model_dir: Path,
    *,
    dry_run: bool = False,
    algorithms: Sequence[str] = rules.ALGORITHMS,
    seeds: Sequence[int] = rules.SEEDS,
    pool_only: list[str] | None = None,
    workers: int = 2,
    progress: Progress = _quiet,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H5a rows (one per trained model) and the H5b table (one per pool model)."""
    log = not dry_run
    if log:
        _setup_mlflow(cfg, rules.MLFLOW_EXPERIMENT)
    split = rules.RQ3_SPLIT
    members = [m for m in pool_members(exp) if pool_only is None or m.model in pool_only]
    progress("H5 checking pool models against step 3.5")
    verify_pool(cfg, exp, model_dir, members)

    tr = train_data(cfg, split)
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
    val = EvalData(x_val.reindex(columns=tr.x.columns), y_val, np.arange(0), np.zeros(0, bool))
    hold_frame = (
        load_part(cfg, split, "val")
        if dry_run
        else load_test(
            cfg,
            rules.HOLDOUT,
            purpose=_purpose(
                "H5a F1 of the trained models and H5b monitoring signals (inputs "
                "only) then H5b F1 decay (labels) of the RQ3 pool models"
            ),
        )
    )
    hold = eval_data(cfg, hold_frame, tr)
    del hold_frame
    late = eval_data(
        cfg,
        open_frame(
            cfg, split, _purpose("H5a secondary: 2017-19 F1 of the trained models"), dry_run
        ),
        tr,
    )

    # H5a: train three arms on 2012-14; measure on validation, 2015-16 and 2017-19.
    rows = []
    for seed in seeds:
        data = {arm: arm_data(tr, arm, seed) for arm in rules.ARMS}
        for algorithm in algorithms:
            for arm in rules.ARMS:
                progress(f"H5a {algorithm} s{seed} {arm}")
                x, y, rejected = data[arm]
                model = build_model(algorithm, seed).fit(x, y)  # type: ignore[arg-type]
                with single_threaded(model):
                    f1_val = detection_metrics(val.y, model.predict_proba(val.x)[:, 1])["f1"]
                h = measure(model, hold, tr, seed, mrs=("MR4",))
                lt = measure(model, late, tr, seed, mrs=())
                m = {
                    "f1_val": f1_val,
                    "f1_holdout": h["f1"],
                    "decay_1516": f1_val - h["f1"],
                    "contrary_acc_holdout": h["contrary_acc"],
                    "rate_MR4_holdout": h["rate_MR4"],
                    "f1_1719": lt["f1"],
                    "decay_1719": f1_val - lt["f1"],
                }
                row: dict[str, Any] = {
                    "split": split,
                    "algorithm": algorithm,
                    "seed": seed,
                    "arm": arm,
                    "n_train": len(y),
                    "n_rejected_copies": rejected,
                    **m,
                }
                row["mlflow_run"] = _log(
                    log,
                    f"H5a-{algorithm}-s{seed}-{arm}",
                    commit,
                    {k: row[k] for k in ("algorithm", "seed", "arm", "n_train")},
                    m,
                )
                rows.append(row)

    # H5b: pool models' label-free signals on the holdout inputs, then decay with its labels.
    ctx = build_context(cfg, exp, model_dir, x_test=hold.x)
    order = sorted(members, key=lambda m: m.model)
    results: list[dict[str, Any]] = []
    if workers <= 1:
        _init(ctx)
        for mem in order:
            results.append(_work(mem))
            progress(f"H5b {mem.model}")
    else:
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=spawn, initializer=_init, initargs=(ctx,)
        ) as ex:
            for mem, res in zip(order, ex.map(_work, order), strict=True):
                results.append(res)
                progress(f"H5b {mem.model}")
    by_model = {r["model"]: r for r in results}
    pre = pd.read_csv(cfg.reports_dir / "rq3" / "pool.csv").set_index("model")
    table = []
    for mem in members:
        r = by_model[mem.model]
        f1_val = detection_metrics(val.y, r["p_val"])["f1"]
        f1_hold = detection_metrics(hold.y, r["p_holdout"])["f1"]
        table.append(
            {
                "model": mem.model,
                "algorithm": mem.algorithm,
                "skew": mem.skew,
                "setting": mem.setting,
                "seed": mem.seed,
                "configuration": mem.configuration,
                "mr4_monitor": r["mr4_monitor"],
                "psi": r["psi"],
                "confidence_drop": r["confidence_drop"],
                "atc_drop": r["atc_drop"],
                "mr4_pre": float(pre.at[mem.model, "mr4_pre"]),  # type: ignore[arg-type]
                "f1_val": f1_val,
                "f1_holdout": f1_hold,
                "decay": f1_val - f1_hold,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(table)
