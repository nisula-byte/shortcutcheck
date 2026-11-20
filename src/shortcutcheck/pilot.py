"""Pilot RQ1 (roadmap step 2.7), exactly as stated in ``docs/pilot_plan.md``.

5 algorithms x 2 feature sets x 5 seeds on one frozen split. Every applicable MR runs on the
whole validation partition with the realism filter on. One MLflow run per model. The Gate 2
criteria are evaluated in code as written in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError
from shortcutcheck.mrs.core import core_mrs
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.mrs.secondary import secondary_mrs
from shortcutcheck.runner.engine import run_mr
from shortcutcheck.runner.stats import MRSummary, summarise
from shortcutcheck.training.baselines import ALL_ALGORITHMS, _setup_mlflow, build_model
from shortcutcheck.training.features import FeatureSet, select

FEATURE_SETS: tuple[FeatureSet, ...] = ("full", "behaviour")
CORE = ("MR1", "MR2", "MR3", "MR4")
CLEAR_LOWER = 0.01  # plan: "clearly non-zero" = 95% CI lower bound above 1% for every seed
MIN_ALGORITHMS = 3


def lower_bound(s: MRSummary) -> float:
    """The CI lower bound the report shows: Wilson at 0 or all violations, else bootstrap."""
    if s.n_counted == 0:
        return float("nan")
    return s.rate_wilson.low if s.n_violations in (0, s.n_counted) else s.rate_ci.low


@dataclass(frozen=True)
class PilotResult:
    split: str
    commit: str
    rows: pd.DataFrame  # one row per (algorithm, feature set, seed, MR, stratum)
    detection: pd.DataFrame  # one row per (algorithm, feature set, seed)


def _mrs(cfg: Config, x_train: pd.DataFrame, y_train: pd.Series) -> list[MetamorphicRelation]:
    valid = (cfg.android_first_year, cfg.collection_year)
    fitted = []
    for mr in [*core_mrs(list(x_train.columns), valid), *secondary_mrs(list(x_train.columns))]:
        try:
            fitted.append(mr.fit(x_train, y_train))
        except NotApplicableError:
            continue
    return fitted


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
        "rate_high": (s.rate_wilson.high if s.n_violations in (0, s.n_counted) else s.rate_ci.high),
        "rate_unfiltered": s.rate_unfiltered,
        "delta": s.delta,
        "delta_low": s.delta_ci.low,
        "delta_high": s.delta_ci.high,
        "max_abs_delta": max_delta,
    }


def run_pilot(cfg: Config, split: str, commit: str) -> PilotResult:
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_train, y_train = select(load_part(cfg, split, "train"), "full", *args)
    x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
    x_val = x_val.reindex(columns=x_train.columns)
    meta = [c for c in cfg.metadata_columns if c in x_train.columns]
    behaviour = [c for c in x_train.columns if c not in cfg.metadata_columns]
    year = x_val["year"].to_numpy(dtype=float)
    valid_year = (year >= cfg.android_first_year) & (year <= cfg.collection_year)

    mrs = _mrs(cfg, x_train, y_train)
    realism = RealismCheck(seed=cfg.seeds[0], metadata_columns=meta).fit(x_train)
    _setup_mlflow(cfg, f"phase2-pilot-{split}")

    rows: list[dict[str, object]] = []
    detection: list[dict[str, object]] = []
    for algorithm in ALL_ALGORITHMS:
        for feature_set in FEATURE_SETS:
            cols = list(x_train.columns) if feature_set == "full" else behaviour
            for seed in cfg.seeds:
                model = build_model(algorithm, seed).fit(x_train[cols], y_train)
                scores = model.predict_proba(x_val[cols])[:, 1]
                quality = detection_metrics(y_val, scores)
                base = {"algorithm": algorithm, "feature_set": feature_set, "seed": seed}
                detection.append({**base, **quality})
                with mlflow.start_run(run_name=f"{algorithm}-{feature_set}-s{seed}") as active:
                    mlflow.set_tags({"step": "2.7", "split": split, "git_commit": commit})
                    mlflow.log_params({**base, "n_train": len(x_train), "n_val": len(x_val)})
                    mlflow.log_metrics(quality)
                    for mr in mrs:
                        run = run_mr(model, mr, x_val, seed=seed, realism=realism)
                        delta = np.abs(run.p_after - run.p_before)[run.counted]
                        max_delta = float(delta.max()) if delta.size else float("nan")
                        info = {**base, "mr": mr.mr_id, "mlflow_run": active.info.run_id}
                        for stratum, mask in (
                            ("all", None),
                            ("valid year", valid_year),
                            ("invalid year", ~valid_year),
                        ):
                            s = summarise(run, mask=mask, seed=seed)
                            rows.append(_row(info, stratum, s, max_delta))
                        overall = summarise(run, seed=seed)
                        mlflow.log_metrics(
                            {
                                f"{mr.mr_id}_rate": overall.rate,
                                f"{mr.mr_id}_rate_low": lower_bound(overall),
                                f"{mr.mr_id}_n_counted": float(overall.n_counted),
                                f"{mr.mr_id}_rate_unfiltered": overall.rate_unfiltered,
                                f"{mr.mr_id}_delta": overall.delta,
                            }
                        )
    return PilotResult(split, commit, pd.DataFrame(rows), pd.DataFrame(detection))


# ---- Gate 2 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate2:
    behaviour_exact_zero: bool
    behaviour_offenders: list[str]
    clear_algorithms: dict[str, list[str]]  # algorithm -> core MRs clearly non-zero

    @property
    def biased_non_zero(self) -> bool:
        return sum(bool(v) for v in self.clear_algorithms.values()) >= MIN_ALGORITHMS

    @property
    def passes(self) -> bool:
        return self.behaviour_exact_zero and self.biased_non_zero


def gate2(result: PilotResult) -> Gate2:
    r = result.rows[result.rows["stratum"] == "all"]
    core = r[r["mr"].isin(CORE)]
    beh = core[core["feature_set"] == "behaviour"]
    bad = beh[(beh["n_violations"] > 0) | (beh["max_abs_delta"].fillna(0) > 0)]
    offenders = sorted({f"{a}/{m}/s{s}" for a, m, s in bad[["algorithm", "mr", "seed"]].values})
    full = core[core["feature_set"] == "full"]
    clear: dict[str, list[str]] = {}
    for algorithm, g in full.groupby("algorithm", sort=False):
        ok = g.groupby("mr")["rate_low"].min()
        clear[str(algorithm)] = sorted(str(m) for m, low in ok.items() if low > CLEAR_LOWER)
    return Gate2(not offenders, offenders, clear)


# ---- report -----------------------------------------------------------------------------


def _f(v: float, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def render_markdown(result: PilotResult) -> str:
    g = gate2(result)
    det = (
        result.detection.groupby(["algorithm", "feature_set"], sort=False)[
            ["f1", "pr_auc", "roc_auc", "fpr_at_95tpr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    lines = [
        f"# Pilot RQ1 on `{result.split}` (roadmap step 2.7)",
        "",
        f"Generated by `shortcutcheck pilot` (`make pilot`). Git commit `{result.commit}`. Plan "
        "and Gate 2 criteria: `docs/pilot_plan.md`, committed before this run. Validation "
        "partition only; test set untouched. Per-model MLflow run IDs are in the CSV next to "
        "this file.",
        "",
        "## Gate 2",
        "",
        "| Criterion | Result |",
        "| --- | --- |",
        "| 1. Known-answer tests pass | yes (step 2.5, `reports/known_answers.md`) |",
        "| 2. Behaviour-only models exactly 0 on MR1–MR4 | "
        + ("**yes**" if g.behaviour_exact_zero else "**no**: " + ", ".join(g.behaviour_offenders))
        + " |",
    ]
    clear = "; ".join(f"{a}: {', '.join(m) or 'none'}" for a, m in g.clear_algorithms.items())
    n_clear = sum(bool(v) for v in g.clear_algorithms.values())
    lines += [
        f"| 3. Full-feature models clearly non-zero (95% CI lower bound > {CLEAR_LOWER:g} for "
        f"all 5 seeds) on a core MR, for at least {MIN_ALGORITHMS} of 5 algorithms "
        f"| {'**yes**' if g.biased_non_zero else '**no**'}: {n_clear} of 5 ({clear}) |",
        "",
        f"**Gate 2 (pilot part): {'PASS' if g.passes else 'FAIL'}.**",
        "",
        "## Detection quality (validation, mean ± SD over 5 seeds)",
        "",
        "| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in det.iterrows():
        cells = [
            f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}"
            for m in ("f1", "pr_auc", "roc_auc", "fpr_at_95tpr")
        ]
        lines.append(
            f"| {row[('algorithm', '')]} | {row[('feature_set', '')]} | " + " | ".join(cells) + " |"
        )
    r = result.rows
    allr = r[r["stratum"] == "all"]
    lines += [
        "",
        "## Violation rates (realism filter on)",
        "",
        "Mean over 5 seeds; in brackets the smallest and largest per-seed 95% CI bound. "
        "*Unfiltered* = same models without the realism filter. Counted = mean inputs counted "
        "per seed.",
        "",
        "| Algorithm | Features | MR | Counted | Violation rate [lowest low, highest high] "
        "| Unfiltered | Mean abs. change in p |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for (a, fs, mr), grp in allr.groupby(["algorithm", "feature_set", "mr"], sort=False):
        lines.append(
            f"| {a} | {fs} | {mr} | {grp['n_counted'].mean():.0f} | {_f(grp['rate'].mean())} "
            f"[{_f(grp['rate_low'].min())}, {_f(grp['rate_high'].max())}] "
            f"| {_f(grp['rate_unfiltered'].mean())} | {_f(grp['delta'].mean())} |"
        )
    strat = r[(r["feature_set"] == "full") & r["mr"].isin(["MR1", "MR4"]) & (r["stratum"] != "all")]
    lines += [
        "",
        "## MR1 and MR4 by year validity (full-feature models; decision D11)",
        "",
        "| Algorithm | MR | Stratum | Counted | Violation rate (mean over seeds) |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for (a, mr, st), grp in strat.groupby(["algorithm", "mr", "stratum"], sort=False):
        lines.append(
            f"| {a} | {mr} | {st} | {grp['n_counted'].mean():.0f} | {_f(grp['rate'].mean())} |"
        )
    return "\n".join(lines) + "\n"
