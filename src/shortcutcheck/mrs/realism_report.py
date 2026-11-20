"""Realism-check evidence on the frozen splits (roadmap step 2.4 Check).

For each split: fit the realism check on the training partition, apply every applicable MR to
a seeded, stratified sample of the validation partition, and report how many perturbed apps
fall beyond the 95th percentile of real-to-real distances. Those apps are excluded from
violation counts.

Violation rates are deliberately **not** reported here. The threshold is fixed before any
violation rate is seen, so the filter cannot be tuned to make results look better. MR4 and MR5
need a model's predicted class, which comes from one random forest (full features, first
seed) trained on the training partition and logged in MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError
from shortcutcheck.mrs.core import core_mrs
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.mrs.secondary import secondary_mrs
from shortcutcheck.runner.engine import run_mr
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select


@dataclass(frozen=True)
class MRRealism:
    mr_id: str
    n_perturbed: int
    n_valid: int
    n_unrealistic: int

    @property
    def share_unrealistic(self) -> float:
        return self.n_unrealistic / self.n_valid if self.n_valid else float("nan")


@dataclass(frozen=True)
class SplitRealism:
    split: str
    n_train: int
    n_eval: int
    threshold: float
    reference_median: float
    real_val_beyond: float
    mlflow_run: str
    rows: list[MRRealism] = field(default_factory=list)


def stratified_sample(y: pd.Series, max_n: int, seed: int) -> np.ndarray:
    """Row positions of a class-stratified sample of at most ``max_n`` rows (all if fewer)."""
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    labels = y.to_numpy()
    picked = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        k = round(max_n * len(idx) / len(labels))
        picked.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def _mrs(columns: list[str], cfg: Config) -> list[MetamorphicRelation]:
    valid = (cfg.android_first_year, cfg.collection_year)
    return [*core_mrs(columns, valid), *secondary_mrs(columns)]


def check_split(
    cfg: Config, split: str, commit: str, max_eval: int = 5000, percentile: float = 95.0
) -> SplitRealism:
    seed = cfg.seeds[0]
    train = load_part(cfg, split, "train")
    val = load_part(cfg, split, "val")
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_train, y_train = select(train, "full", *args)
    x_val, y_val = select(val, "full", *args)
    x_val = x_val.reindex(columns=x_train.columns)
    pick = stratified_sample(y_val, max_eval, seed)
    x_eval = x_val.iloc[pick].reset_index(drop=True)

    realism = RealismCheck(percentile=percentile, seed=seed).fit(x_train)
    real_beyond = float((~realism.is_realistic(x_eval)).mean())
    model = build_model("rf", seed).fit(x_train, y_train)

    rows = []
    for mr in _mrs(list(x_train.columns), cfg):
        try:
            mr.fit(x_train, y_train)
        except NotApplicableError:
            continue
        run = run_mr(model, mr, x_eval, seed=seed, realism=realism)
        n_valid = int(run.counted_unfiltered.sum())
        rows.append(MRRealism(mr.mr_id, int(run.perturbed.sum()), n_valid, run.n_unrealistic))

    _setup_mlflow(cfg, f"phase2-realism-{split}")
    with mlflow.start_run(run_name=f"realism-rf-s{seed}") as active:
        mlflow.set_tags({"step": "2.4", "split": split, "git_commit": commit})
        mlflow.log_params(
            {
                "percentile": percentile,
                "reference_sample": realism.reference_sample,
                "max_eval": max_eval,
                "seed": seed,
                "model": "rf-full",
            }
        )
        metrics = {
            "threshold": realism.threshold,
            "real_val_beyond": real_beyond,
            "n_eval": float(len(x_eval)),
        }
        for r in rows:
            metrics[f"{r.mr_id}_n_valid"] = float(r.n_valid)
            metrics[f"{r.mr_id}_n_unrealistic"] = float(r.n_unrealistic)
        mlflow.log_metrics(metrics)
        run_id = active.info.run_id

    return SplitRealism(
        split=split,
        n_train=len(x_train),
        n_eval=len(x_eval),
        threshold=realism.threshold,
        reference_median=float(np.median(realism.reference_distances)),
        real_val_beyond=real_beyond,
        mlflow_run=run_id,
        rows=rows,
    )


def _pct(v: float) -> str:
    return "–" if np.isnan(v) else f"{v:.1%}"


def render_markdown(results: list[SplitRealism], commit: str, percentile: float = 95.0) -> str:
    ids = sorted({r.mr_id for s in results for r in s.rows})
    lines = [
        "# Realism check (roadmap step 2.4)",
        "",
        "Generated by `shortcutcheck mrs realism`. Git commit " f"`{commit}`. Do not edit by hand.",
        "",
        "Distance: Gower distance to the nearest real training app (0/1 flags compared by "
        "mismatch, numeric features by training percentile). Threshold: the "
        f"{percentile:g}th percentile of distances from a training app to its nearest *other* "
        "training app (2,000-app seeded sample). Perturbed apps beyond the threshold are "
        "excluded from violation counts. Evaluation inputs: a seeded, class-stratified sample "
        "of up to 5,000 validation apps. Violation rates are not shown here, so the filter is "
        "fixed before any violation rate is seen.",
        "",
        "## Threshold and calibration",
        "",
        "If the training sample represents real apps well, about "
        f"{100 - percentile:g}% of *unperturbed* validation apps should fall beyond the "
        "threshold.",
        "",
        "| Split | Training apps | Evaluated apps | Median real-to-real distance | Threshold "
        "| Unperturbed validation apps beyond threshold | MLflow run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in results:
        lines.append(
            f"| {s.split} | {s.n_train} | {s.n_eval} | {s.reference_median:.4f} "
            f"| {s.threshold:.4f} | {_pct(s.real_val_beyond)} | `{s.mlflow_run}` |"
        )
    lines += [
        "",
        "## Perturbed apps beyond the threshold (excluded)",
        "",
        "Each cell: excluded / perturbed-and-valid (share). A dash means the MR does not apply "
        "to that split (see `reports/mr_applicability.md`).",
        "",
        "| Split | " + " | ".join(ids) + " |",
        "| --- | " + " | ".join("---:" for _ in ids) + " |",
    ]
    for s in results:
        by = {r.mr_id: r for r in s.rows}
        cells = [
            (
                f"{by[i].n_unrealistic} / {by[i].n_valid} ({_pct(by[i].share_unrealistic)})"
                if i in by
                else "–"
            )
            for i in ids
        ]
        lines.append(f"| {s.split} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
