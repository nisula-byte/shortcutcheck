"""H1 (RQ1, roadmap step 3.3), judged exactly as PREREGISTRATION.md states it.

Input: ``reports/rq1/rows.csv`` from ``make rq1``. Only the ``all`` stratum, the realism-filtered
rate and its per-seed 95% CI lower bound are used.

* An algorithm **shows the shortcut** on a split if, for at least one core MR, its mean
  violation rate over the 5 seeds is at least ``RATE_MIN`` (5%, the proposal's target, unchanged)
  and the CI lower bound is above ``LOW_MIN`` (1%) for every one of the 5 seeds.
* **H1a** holds on a split if at least ``MIN_ALGORITHMS`` (3) of the 5 algorithms show it.
* **Control**: behaviour-only models must be exactly 0 on every core MR (no violation and no
  change in p). They cannot see metadata, so this holds by construction; a failure means the
  runner is broken, and then H1 is not judged at all.
* **Verdict**: supported if H1a holds on both biased splits, partially supported on one,
  not supported on neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

CORE_MRS = ("MR1", "MR2", "MR3", "MR4")
BIASED_SPLITS = ("rq12-DR-AG", "rq12-VS-AG")
RATE_MIN = 0.05
LOW_MIN = 0.01
MIN_ALGORITHMS = 3
N_SEEDS = 5


@dataclass(frozen=True)
class SplitJudgement:
    split: str
    table: pd.DataFrame  # algorithm x MR: mean, sd, min_low, n_seeds, shows
    algorithms_showing: list[str]

    @property
    def h1a(self) -> bool:
        return len(self.algorithms_showing) >= MIN_ALGORITHMS


@dataclass(frozen=True)
class H1Result:
    splits: list[SplitJudgement]
    control_offenders: list[str]

    @property
    def control_ok(self) -> bool:
        return not self.control_offenders

    @property
    def verdict(self) -> str:
        if not self.control_ok:
            return "NOT JUDGED (behaviour-only control failed: runner error)"
        held = sum(s.h1a for s in self.splits)
        if held == len(BIASED_SPLITS):
            return "SUPPORTED"
        return "PARTIALLY SUPPORTED" if held else "NOT SUPPORTED"


def _own(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[(rows["stratum"] == "all") & rows["mr"].isin(CORE_MRS)]
    if "role" in r.columns:
        r = r[r["role"] == "own"]
    return r


def judge_split(rows: pd.DataFrame, split: str) -> SplitJudgement:
    r = _own(rows)
    r = r[(r["train_split"] == split) & (r["feature_set"] == "full")]
    table = (
        r.groupby(["algorithm", "mr"], sort=False)
        .agg(
            mean=("rate", "mean"),
            sd=("rate", "std"),
            min_low=("rate_low", "min"),
            n_seeds=("rate", "count"),  # non-missing rates only
            all_low_ok=("rate_low", lambda s: bool((s > LOW_MIN).all())),
        )
        .reset_index()
    )
    table["shows"] = (
        (table["n_seeds"] == N_SEEDS) & (table["mean"] >= RATE_MIN) & table["all_low_ok"]
    )
    showing = sorted(table.loc[table["shows"], "algorithm"].astype(str).unique().tolist())
    return SplitJudgement(split, table.drop(columns="all_low_ok"), showing)


def control_offenders(rows: pd.DataFrame) -> list[str]:
    r = _own(rows)
    beh = r[r["feature_set"] == "behaviour"]
    bad = beh[(beh["n_violations"] > 0) | (beh["max_abs_delta"].fillna(0) > 0)]
    return sorted(
        {
            f"{s}/{a}/{m}/s{seed}"
            for s, a, m, seed in bad[["train_split", "algorithm", "mr", "seed"]].values
        }
    )


def judge_h1(rows: pd.DataFrame) -> H1Result:
    return H1Result([judge_split(rows, s) for s in BIASED_SPLITS], control_offenders(rows))


def _f(v: Any, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def render_h1(result: H1Result, rows: pd.DataFrame, detection: pd.DataFrame, header: str) -> str:
    lines = [
        "# RQ1: does MR testing detect shortcut reliance? (H1, roadmap step 3.3)",
        "",
        header,
        "",
        f"**H1: {result.verdict}.**",
        "",
        "| Split | Algorithms showing the shortcut (mean ≥ "
        f"{RATE_MIN:g} and CI lower bound > {LOW_MIN:g} for all {N_SEEDS} seeds, on a core MR) "
        f"| H1a (≥ {MIN_ALGORITHMS} of 5) |",
        "| --- | --- | --- |",
    ]
    for s in result.splits:
        names = ", ".join(s.algorithms_showing) or "none"
        held = "holds" if s.h1a else "does not hold"
        lines.append(f"| {s.split} | {len(s.algorithms_showing)}: {names} | {held} |")
    lines += [
        "",
        "Control (behaviour-only models exactly 0 on every core MR): "
        + (
            "**passes**."
            if result.control_ok
            else "**FAILS**: " + ", ".join(result.control_offenders)
        ),
        "",
        "## Core MRs, full-feature models",
        "",
        "| Split | Algorithm | MR | Seeds | Mean rate ± SD | Lowest CI lower bound | Shows |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for s in result.splits:
        for t in s.table.itertuples():
            lines.append(
                f"| {s.split} | {t.algorithm} | {t.mr} | {t.n_seeds} | {_f(t.mean)} ± {_f(t.sd)} "
                f"| {_f(t.min_low)} | {'yes' if t.shows else 'no'} |"
            )
    r = rows[rows["stratum"] == "all"]
    lines += [
        "",
        "## Every MR, both feature sets (descriptive)",
        "",
        "Mean over seeds. *Unfiltered* = without the realism filter. MR5 and MR6 are secondary.",
        "",
        "| Split | Algorithm | Features | MR | Counted | Rate | Unfiltered "
        "| Mean abs. change in p |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (sp, a, fs, mr), g in r.groupby(
        ["train_split", "algorithm", "feature_set", "mr"], sort=False
    ):
        lines.append(
            f"| {sp} | {a} | {fs} | {mr} | {g['n_counted'].mean():.0f} | {_f(g['rate'].mean())} "
            f"| {_f(g['rate_unfiltered'].mean())} | {_f(g['delta'].mean())} |"
        )
    strat = rows[(rows["feature_set"] == "full") & rows["mr"].isin(["MR1", "MR4"])]
    strat = strat[strat["stratum"] != "all"]
    lines += [
        "",
        "## MR1 and MR4 by year validity (D11; descriptive)",
        "",
        "| Split | Algorithm | MR | Stratum | Counted | Rate |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for (sp, a, mr, st), g in strat.groupby(
        ["train_split", "algorithm", "mr", "stratum"], sort=False
    ):
        lines.append(
            f"| {sp} | {a} | {mr} | {st} | {g['n_counted'].mean():.0f} | {_f(g['rate'].mean())} |"
        )
    det = (
        detection.groupby(["train_split", "algorithm", "feature_set"], sort=False)[
            ["f1", "pr_auc", "roc_auc", "fpr_at_95tpr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    lines += [
        "",
        "## Detection quality on the test partition (mean ± SD over seeds)",
        "",
        "| Split | Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in det.iterrows():
        cells = [
            f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}"
            for m in ("f1", "pr_auc", "roc_auc", "fpr_at_95tpr")
        ]
        lines.append(
            f"| {row[('train_split', '')]} | {row[('algorithm', '')]} | "
            f"{row[('feature_set', '')]} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"
