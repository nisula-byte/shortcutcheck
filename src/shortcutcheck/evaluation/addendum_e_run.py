"""Addendum E run: the AZL replication of H5a/H5b/H5c, the adjusted H5b test and the
behaviour-only comparison.

Data access (PREREGISTRATION-ADDENDUM-E.md): AZL 2015-16, AZL 2017-19 and AndroCT 2015-16 are
each opened once through ``load_test()`` with the purpose logged. Labels are separated from the
inputs straight away; label-free signals see inputs only. With ``dry_run=True`` the validation
partition of the training split replaces every test part, nothing is logged as a test access,
and the output is not a result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from shortcutcheck import signals as sig
from shortcutcheck.addenda import addendum_d as d
from shortcutcheck.addenda import addendum_e as rules
from shortcutcheck.config import Config
from shortcutcheck.data.dada import split_feature_groups
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.addendum_d_run import EvalData, TrainData, arm_data, eval_data
from shortcutcheck.evaluation.addendum_d_run import train_data as d_train_data
from shortcutcheck.evaluation.locked_test import load_test
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.runner.engine import malware_probability, run_mr
from shortcutcheck.runner.stats import summarise
from shortcutcheck.training.baselines import _setup_mlflow, build_model

Progress = Callable[[str], None]
RQ3_CONFIG = Path("configs/experiments/rq3.yaml")


def _quiet(_m: str) -> None:
    return None


def _purpose(what: str) -> str:
    return f"addendum E (PREREGISTRATION-ADDENDUM-E.md): {what}"


def rq3_settings(root: Path) -> tuple[dict[str, list[dict[str, Any]]], int, float]:
    """Hyperparameter settings and PSI settings of the RQ3 pool (configs/experiments/rq3.yaml)."""
    raw = yaml.safe_load((root / RQ3_CONFIG).read_text(encoding="utf-8"))
    hyper = {str(k): [dict(s or {}) for s in v] for k, v in raw["pool"]["hyperparameters"].items()}
    psi = raw["pool"]["signals"]["monitoring"]["psi"]
    return hyper, int(psi["bins"]), float(psi["epsilon"])


def arm(
    tr: TrainData, name: str, seed: int, metadata: list[str]
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Training data of one arm; 'behaviour' drops every metadata column."""
    if name == "behaviour":
        _, behaviour = split_feature_groups(list(tr.x.columns), metadata)
        return tr.x[behaviour], tr.y, 0
    return arm_data(tr, name, seed)


def open_test(cfg: Config, split: str, val_of: str, purpose: str, dry_run: bool) -> pd.DataFrame:
    if dry_run:
        return load_part(cfg, val_of, "val")
    return load_test(cfg, split, purpose=purpose)


def f1(model: Any, ev: EvalData) -> float:
    return float(detection_metrics(ev.y, malware_probability(model, ev.x))["f1"])


def mr4_rate(model: Any, ev: EvalData, tr: TrainData, seed: int) -> float:
    xs = ev.x.iloc[ev.sample].reset_index(drop=True)
    return float(summarise(run_mr(model, tr.mrs["MR4"], xs, seed=seed, realism=tr.realism)).rate)


def _log(
    log: bool, name: str, commit: str, params: dict[str, Any], metrics: dict[str, float]
) -> str:
    if not log:
        return ""
    with mlflow.start_run(run_name=name) as r:
        mlflow.set_tags({"git_commit": commit, "addendum": "E"})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not pd.isna(v)})
        return str(r.info.run_id)


def arms_rows(
    tr: TrainData,
    val: EvalData,
    hold: EvalData,
    late: EvalData | None,
    metadata: list[str],
    arms: Sequence[str],
    algorithms: Sequence[str],
    seeds: Sequence[int],
    split: str,
    commit: str,
    log: bool,
    progress: Progress,
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        data = {a: arm(tr, a, seed, metadata) for a in arms}
        for algorithm in algorithms:
            for a in arms:
                progress(f"{split} {algorithm} s{seed} {a}")
                x, y, rejected = data[a]
                model = build_model(algorithm, seed).fit(x, y)  # type: ignore[arg-type]
                f1_val, f1_hold = f1(model, val), f1(model, hold)
                m = {
                    "f1_val": f1_val,
                    "f1_holdout": f1_hold,
                    "decay_1516": f1_val - f1_hold,
                    "rate_MR4_holdout": mr4_rate(model, hold, tr, seed),
                }
                if late is not None:
                    m["f1_1719"] = f1(model, late)
                    m["decay_1719"] = f1_val - m["f1_1719"]
                row: dict[str, Any] = {
                    "split": split,
                    "algorithm": algorithm,
                    "seed": seed,
                    "arm": a,
                    "n_train": len(y),
                    "n_rejected_copies": rejected,
                    **m,
                }
                row["mlflow_run"] = _log(
                    log,
                    f"E-{split}-{algorithm}-s{seed}-{a}",
                    commit,
                    {k: row[k] for k in ("split", "algorithm", "seed", "arm", "n_train")},
                    m,
                )
                rows.append(row)
    return rows


def pool_rows(
    tr: TrainData,
    val: EvalData,
    hold: EvalData,
    hyper: dict[str, list[dict[str, Any]]],
    bins: int,
    floor: float,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    settings: int,
    progress: Progress,
) -> list[dict[str, Any]]:
    """Label-free signals on the deployment inputs first, then F1 decay with the labels."""
    sig.assert_unlabelled(hold.x)
    rows = []
    for algorithm in algorithms:
        for setting in range(settings):
            for seed in seeds:
                progress(f"pool {algorithm} h{setting} s{seed}")
                params = hyper[algorithm][setting] or None
                model = build_model(algorithm, seed, params=params).fit(  # type: ignore[arg-type]
                    tr.x, tr.y
                )
                p_val = malware_probability(model, val.x)
                p_new = malware_probability(model, hold.x)
                correct = (p_val >= 0.5).astype(int) == val.y.to_numpy().astype(int)
                threshold = sig.atc_threshold(sig.confidence(p_val), correct)
                signals = {
                    "mr4_monitor": mr4_rate(model, hold, tr, seed),
                    "psi": sig.psi(p_val, p_new, bins, floor),
                    "confidence_drop": sig.confidence_drop(p_val, p_new),
                    "atc_drop": sig.atc_drop(
                        float(correct.mean()), threshold, sig.confidence(p_new)
                    ),
                }
                f1_val = float(detection_metrics(val.y, p_val)["f1"])
                f1_hold = float(detection_metrics(hold.y, p_new)["f1"])
                rows.append(
                    {
                        "model": f"{algorithm}-h{setting}-s{seed}",
                        "algorithm": algorithm,
                        "setting": setting,
                        "seed": seed,
                        **signals,
                        "f1_val": f1_val,
                        "f1_holdout": f1_hold,
                        "decay": f1_val - f1_hold,
                    }
                )
    return rows


def run_e(
    cfg: Config,
    commit: str,
    *,
    dry_run: bool = False,
    algorithms: Sequence[str] = d.ALGORITHMS,
    seeds: Sequence[int] = d.SEEDS,
    settings: int = rules.SETTINGS,
    androct_secondary: bool = True,
    progress: Progress = _quiet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """AZL arms, AZL pool and the AndroCT behaviour-only rows (secondary)."""
    log = not dry_run
    if log:
        _setup_mlflow(cfg, rules.MLFLOW_EXPERIMENT)
    hyper, bins, floor = rq3_settings(cfg.root)
    tr = d_train_data(cfg, rules.AZL_SPLIT)
    metadata = [c for c in cfg.metadata_columns if c in tr.x.columns]
    val = eval_data(cfg, load_part(cfg, rules.AZL_SPLIT, "val"), tr)
    hold = eval_data(
        cfg,
        open_test(
            cfg,
            rules.AZL_HOLDOUT,
            rules.AZL_SPLIT,
            _purpose(
                "E1/E5 F1 of the trained arms; E2-E4 pool signals (inputs only) then F1 "
                "decay (labels)"
            ),
            dry_run,
        ),
        tr,
    )
    late = eval_data(
        cfg,
        open_test(
            cfg,
            rules.AZL_SPLIT,
            rules.AZL_SPLIT,
            _purpose("reported 2017-19 F1 of the trained arms"),
            dry_run,
        ),
        tr,
    )
    pool = pool_rows(tr, val, hold, hyper, bins, floor, algorithms, seeds, settings, progress)
    arms = arms_rows(
        tr,
        val,
        hold,
        late,
        metadata,
        rules.ARMS,
        algorithms,
        seeds,
        rules.AZL_SPLIT,
        commit,
        log,
        progress,
    )
    androct: list[dict[str, Any]] = []
    if androct_secondary:
        tr2 = d_train_data(cfg, d.RQ3_SPLIT)
        val2 = eval_data(cfg, load_part(cfg, d.RQ3_SPLIT, "val"), tr2)
        hold2 = eval_data(
            cfg,
            open_test(
                cfg,
                d.HOLDOUT,
                d.RQ3_SPLIT,
                _purpose("secondary E5: F1 of 25 new behaviour-only AndroCT models"),
                dry_run,
            ),
            tr2,
        )
        androct = arms_rows(
            tr2,
            val2,
            hold2,
            None,
            metadata,
            ("behaviour",),
            algorithms,
            seeds,
            d.RQ3_SPLIT,
            commit,
            log,
            progress,
        )
    return pd.DataFrame(arms), pd.DataFrame(pool), pd.DataFrame(androct)


def androct_e5_rows(h5a_rows: pd.DataFrame, behaviour_rows: pd.DataFrame) -> pd.DataFrame:
    """Addendum D's augmented AndroCT rows next to the new behaviour-only rows."""
    keep = ["split", "algorithm", "seed", "arm", "f1_val", "f1_holdout", "decay_1516"]
    aug = h5a_rows[h5a_rows["arm"] == "augmented"][keep]
    return pd.concat([aug, behaviour_rows[keep]], ignore_index=True)
