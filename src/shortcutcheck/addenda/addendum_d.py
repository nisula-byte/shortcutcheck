"""Addendum D (PREREGISTRATION-ADDENDUM-D.md): decision rules and reports for H4 and H5.

This module only judges saved tables. The data are read by ``evaluation.addendum_d_run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shortcutcheck.analysis.h3 import Correlations, correlations
from shortcutcheck.analysis.stats import bootstrap_p_greater, holm, percentile_ci, wilcoxon_greater

FILE = "PREREGISTRATION-ADDENDUM-D.md"
MLFLOW_EXPERIMENT = "addendum-d-confirm"
H4_SPLITS = ("rq12-DR-AG", "rq12-DR-AG_deb", "rq12-VS-AG", "rq12-VS-AG_deb")
NO_CONTROL = ("rq12-VS-AG",)
RQ3_SPLIT = "rq3-AndroCT-2012-14-vs-2017-19"
HOLDOUT = "rq3-AndroCT-holdout-2015-16"
ALGORITHMS = ("lr", "svm", "rf", "lgbm", "mlp")
SEEDS = (11, 22, 33, 44, 55)
EVAL_MRS = ("MR1", "MR3", "MR4", "MR5", "MR6")
HELD_OUT = ("MR1", "MR3")
ARMS = ("baseline", "augmented", "duplicate")
MR_SAMPLE = 5000
SAMPLE_SEED = 11
ALPHA = 0.05
RATE_RATIO = 0.5
F1_DROP = 0.02
MIN_PASSING = 3
RHO_MIN = 0.5
RESAMPLES = 2000
BOOT_SEED = 11
H5B_PRIMARY = "mr4_monitor"
H5B_COMPETITORS = ("psi", "confidence_drop", "atc_drop")
PAIR = ["split", "algorithm", "seed"]


def paired(
    rows: pd.DataFrame, measure: str, a: str = "baseline", b: str = "augmented"
) -> pd.DataFrame:
    """Two arms of one measure side by side, matched by split, algorithm and seed."""
    wide = rows.pivot_table(index=PAIR, columns="arm", values=measure, aggfunc="first")
    for arm in (a, b):
        if arm not in wide.columns:
            wide[arm] = np.nan
    return wide[[a, b]].reset_index().dropna()


def _p_greater(d: np.ndarray) -> float:
    return wilcoxon_greater(d.tolist()).p_greater if len(d) else float("nan")


# ---- H4 -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class H4Split:
    split: str
    pairs: int
    mr4_baseline: float
    mr4_augmented: float
    p_mr4_holm: float
    f1_baseline: float
    f1_augmented: float
    p_held_out_holm: float
    contrary_change: float
    p_contrary_holm: float

    @property
    def a(self) -> bool:
        return bool(
            self.p_mr4_holm < ALPHA and self.mr4_augmented <= RATE_RATIO * self.mr4_baseline
        )

    @property
    def b(self) -> bool:
        return bool(self.f1_baseline - self.f1_augmented <= F1_DROP)

    @property
    def c(self) -> bool:
        return bool(self.p_held_out_holm < ALPHA)

    @property
    def d(self) -> bool:
        return bool(self.p_contrary_holm < ALPHA)

    @property
    def passes(self) -> bool:
        return self.a and self.b and self.c


@dataclass(frozen=True)
class H4Result:
    splits: list[H4Split]
    held_out: pd.DataFrame

    @property
    def n_passing(self) -> int:
        return sum(s.passes for s in self.splits)

    @property
    def verdict(self) -> str:
        if self.n_passing >= MIN_PASSING:
            return "SUPPORTED"
        return "PARTIALLY SUPPORTED" if self.n_passing >= 1 else "NOT SUPPORTED"

    @property
    def n_h4d(self) -> int:
        return sum(s.d for s in self.splits)


def held_out_tests(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for split in dict.fromkeys(rows["split"]):
        for mr in HELD_OUT:
            p = paired(rows[rows["split"] == split], f"rate_{mr}")
            if p.empty:
                continue
            d = (p["baseline"] - p["augmented"]).to_numpy()
            out.append(
                {
                    "split": split,
                    "mr": mr,
                    "pairs": len(d),
                    "mean_baseline": float(p["baseline"].mean()),
                    "mean_augmented": float(p["augmented"].mean()),
                    "p": _p_greater(d),
                }
            )
    cols = ["split", "mr", "pairs", "mean_baseline", "mean_augmented", "p"]
    frame = pd.DataFrame(out, columns=cols)
    frame["p_holm"] = holm(frame["p"].tolist()) if len(frame) else []
    return frame


def judge_h4(rows: pd.DataFrame) -> H4Result:
    splits = list(dict.fromkeys(rows["split"]))
    mr4 = {s: paired(rows[rows["split"] == s], "rate_MR4") for s in splits}
    con = {s: paired(rows[rows["split"] == s], "contrary_acc") for s in splits}
    p_mr4 = holm([_p_greater((m["baseline"] - m["augmented"]).to_numpy()) for m in mr4.values()])
    p_con = holm([_p_greater((c["augmented"] - c["baseline"]).to_numpy()) for c in con.values()])
    tests = held_out_tests(rows)
    out = []
    for i, s in enumerate(splits):
        f1 = paired(rows[rows["split"] == s], "f1")
        t = tests[tests["split"] == s]
        out.append(
            H4Split(
                s,
                len(mr4[s]),
                float(mr4[s]["baseline"].mean()),
                float(mr4[s]["augmented"].mean()),
                float(p_mr4[i]),
                float(f1["baseline"].mean()),
                float(f1["augmented"].mean()),
                float(t["p_holm"].min()) if len(t) else float("nan"),
                float((con[s]["augmented"] - con[s]["baseline"]).mean()),
                float(p_con[i]),
            )
        )
    return H4Result(out, tests)


# ---- H5a ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class H5aResult:
    pairs: int
    decay_baseline: float
    decay_augmented: float
    p: float

    @property
    def holds(self) -> bool:
        return bool(self.p < ALPHA)


def judge_h5a(rows: pd.DataFrame, decay: str = "decay_1516") -> H5aResult:
    p = paired(rows, decay)
    d = (p["baseline"] - p["augmented"]).to_numpy()
    return H5aResult(
        len(d), float(p["baseline"].mean()), float(p["augmented"].mean()), _p_greater(d)
    )


# ---- H5b and H5c --------------------------------------------------------------------------


@dataclass(frozen=True)
class H5bResult:
    headline: Correlations
    full: Correlations
    comparisons: pd.DataFrame
    pre: dict[str, Correlations]

    @property
    def h5b(self) -> bool:
        rho, low = self.headline.rho[H5B_PRIMARY], self.headline.ci[H5B_PRIMARY][0]
        return bool(rho >= RHO_MIN and low > 0)

    def h5c(self) -> dict[str, bool]:
        return {
            str(r["competitor"]): bool(r["p_holm"] < ALPHA)
            for r in self.comparisons.to_dict("records")
        }


def judge_h5b(table: pd.DataFrame, resamples: int = RESAMPLES, seed: int = BOOT_SEED) -> H5bResult:
    signals = (H5B_PRIMARY, *H5B_COMPETITORS)
    complete = table.dropna(subset=[*signals, "decay"])
    headline = correlations(
        complete[complete["skew"] == 0], "un-injected", signals, "decay", resamples, seed
    )
    full = correlations(complete, "full pool", signals, "decay", resamples, seed)
    out = []
    for k in H5B_COMPETITORS:
        diff = (headline.boot[H5B_PRIMARY] - headline.boot[k]).to_numpy()
        low, high = percentile_ci(diff)
        out.append(
            {
                "competitor": k,
                "rho_mr4_monitor": headline.rho[H5B_PRIMARY],
                "rho_competitor": headline.rho[k],
                "delta": headline.rho[H5B_PRIMARY] - headline.rho[k],
                "delta_low": low,
                "delta_high": high,
                "p_greater": bootstrap_p_greater(diff),
            }
        )
    comp = pd.DataFrame(out)
    comp["p_holm"] = holm(comp["p_greater"].tolist())
    pre: dict[str, Correlations] = {}
    if "mr4_pre" in complete.columns:
        for subset, part in (
            ("un-injected", complete[complete["skew"] == 0]),
            ("full pool", complete),
        ):
            pre[subset] = correlations(part, subset, ("mr4_pre",), "decay", resamples, seed)
    return H5bResult(headline, full, comp, pre)


# ---- reports ------------------------------------------------------------------------------


def _f(v: Any) -> str:
    v = float(v)
    if np.isnan(v):
        return "n/a"
    return f"{v:.1e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}"


def _yes(b: bool) -> str:
    return "yes" if b else "no"


def header(command: str, commit: str, prereg: str) -> str:
    return (
        f"Generated by `shortcutcheck addendum d {command}` at commit `{commit}`. Rules: "
        f"`{FILE}` (committed in `{prereg[:7]}` before any data it tests was opened). It does "
        "not change H1-H3 or addenda A-C."
    )


def arm_means(rows: pd.DataFrame, measures: list[str]) -> list[str]:
    present = [m for m in measures if m in rows.columns]
    means = rows.groupby(["split", "arm"], sort=False)[present].mean().reset_index()
    lines = [
        "| Split | Arm | " + " | ".join(present) + " |",
        "| --- | --- |" + " ---: |" * len(present),
    ]
    for r in means.to_dict("records"):
        lines.append(
            f"| {r['split']} | {r['arm']} | " + " | ".join(_f(r[m]) for m in present) + " |"
        )
    return lines


def render_h4(rows: pd.DataFrame, r: H4Result, head: str) -> str:
    lines = [
        "# Addendum D, RQ4 / H4: does the fix hold on unseen test inputs?",
        "",
        head,
        "",
        f"**H4: {r.verdict}** ({r.n_passing} of {len(r.splits)} splits pass H4a, H4b and H4c; "
        f"SUPPORTED needs {MIN_PASSING}). **H4d** (contrary-app accuracy rises, Holm): holds in "
        f"{r.n_h4d} of {len(r.splits)} splits.",
        "",
        "| Split | Pairs | MR4 baseline → augmented | H4a Holm p | H4a | F1 baseline → augmented "
        "| H4b | H4c min Holm p | H4c | Passes | Contrary acc. change | H4d Holm p |",
        "| --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for s in r.splits:
        lines.append(
            f"| {s.split} | {s.pairs} | {_f(s.mr4_baseline)} → {_f(s.mr4_augmented)} | "
            f"{_f(s.p_mr4_holm)} | {_yes(s.a)} | {_f(s.f1_baseline)} → {_f(s.f1_augmented)} | "
            f"{_yes(s.b)} | {_f(s.p_held_out_holm)} | {_yes(s.c)} | **{_yes(s.passes)}** | "
            f"{s.contrary_change:+.3f} | {_f(s.p_contrary_holm)} |"
        )
    lines += [
        "",
        "## H4c: held-out metadata MRs",
        "",
        "| Split | MR | Pairs | Mean baseline | Mean augmented | p | p (Holm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in r.held_out.to_dict("records"):
        lines.append(
            f"| {t['split']} | {t['mr']} | {t['pairs']} | {_f(t['mean_baseline'])} | "
            f"{_f(t['mean_augmented'])} | {_f(t['p'])} | {_f(t['p_holm'])} |"
        )
    lines += [
        "",
        "## Reported, not judged: every arm (means over models)",
        "",
        "The duplicate control trains on every app twice, unchanged (not run on rq12-VS-AG). "
        "MR5 and MR6 change permissions, not metadata (D20).",
        "",
        *arm_means(
            rows,
            ["f1", "roc_auc", "contrary_acc", *(f"rate_{m}" for m in EVAL_MRS)],
        ),
        "",
        "## Per algorithm (MR4 and F1, mean over seeds)",
        "",
        "| Split | Algorithm | MR4 baseline | MR4 augmented | F1 baseline | F1 augmented |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    per = rows[rows["arm"].isin(["baseline", "augmented"])]
    g = per.groupby(["split", "algorithm", "arm"], sort=False)[["rate_MR4", "f1"]].mean()
    for (split, alg), part in g.groupby(level=[0, 1], sort=False):
        v = part.droplevel([0, 1])
        lines.append(
            f"| {split} | {alg} | {_f(v.loc['baseline', 'rate_MR4'])} | "
            f"{_f(v.loc['augmented', 'rate_MR4'])} | {_f(v.loc['baseline', 'f1'])} | "
            f"{_f(v.loc['augmented', 'f1'])} |"
        )
    return "\n".join(lines) + "\n"


def render_h5(rows: pd.DataFrame, a: H5aResult, b: H5bResult, head: str) -> str:
    a17 = judge_h5a(rows, "decay_1719")
    dup = paired(rows, "decay_1516", "duplicate", "augmented")
    d_dup = (dup["duplicate"] - dup["augmented"]).to_numpy()
    lines = [
        "# Addendum D, H5: forward-looking tests on the locked 2015-16 holdout",
        "",
        head,
        "",
        f"- **H5a (the fix reduces decay): {'HOLDS' if a.holds else 'DOES NOT HOLD'}.** Mean decay "
        f"F1(2012-14 validation) − F1(2015-16): baseline {_f(a.decay_baseline)}, augmented "
        f"{_f(a.decay_augmented)}; one-sided Wilcoxon over {a.pairs} pairs p = {_f(a.p)}.",
        f"- **H5b (MR4 at monitoring time predicts decay): "
        f"{'HOLDS' if b.h5b else 'DOES NOT HOLD'}.** Un-injected models: ρ = "
        f"{_f(b.headline.rho[H5B_PRIMARY])} (95% CI {_f(b.headline.ci[H5B_PRIMARY][0])} to "
        f"{_f(b.headline.ci[H5B_PRIMARY][1])}); rule ρ ≥ {RHO_MIN:g} and CI lower bound > 0.",
        "- **H5c (MR4 at monitoring time beats each baseline):** "
        + ", ".join(f"{k} {'holds' if v else 'does not hold'}" for k, v in b.h5c().items())
        + ".",
        "",
        "## H5a: decay per arm (reported, not judged beyond the rule above)",
        "",
        *arm_means(
            rows,
            ["f1_val", "f1_holdout", "decay_1516", "f1_1719", "decay_1719", "rate_MR4_holdout"],
        ),
        "",
        f"- Same comparison on 2017-19 (reported): baseline {_f(a17.decay_baseline)}, augmented "
        f"{_f(a17.decay_augmented)}, one-sided p = {_f(a17.p)}.",
        f"- Augmented vs duplicate control (2015-16 decay, reported): mean difference "
        f"{(d_dup.mean() if len(d_dup) else float('nan')):+.3f}, one-sided p = "
        f"{_f(_p_greater(d_dup))}.",
        "",
        "## H5b and H5c: Spearman ρ with 2015-16 decay",
        "",
        f"Cluster bootstrap over configurations ({RESAMPLES} resamples, seed {BOOT_SEED}).",
        "",
        "| Subset | Models | Configurations | Signal | ρ | 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for c in (b.headline, b.full, *b.pre.values()):
        for s, rho in c.rho.items():
            lines.append(
                f"| {c.subset} | {c.n_models} | {c.n_configurations} | {s} | {_f(rho)} | "
                f"{_f(c.ci[s][0])} to {_f(c.ci[s][1])} |"
            )
    lines += [
        "",
        "`mr4_pre` (pre-deployment, from step 3.6) re-tests H3 on a second deployment period "
        "(reported, not judged).",
        "",
        "| H5c: MR4 monitor vs | ρ MR4 monitor | ρ competitor | Δρ | 95% CI | p | p (Holm) |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for r in b.comparisons.to_dict("records"):
        lines.append(
            f"| {r['competitor']} | {_f(r['rho_mr4_monitor'])} | {_f(r['rho_competitor'])} | "
            f"{_f(r['delta'])} | {_f(r['delta_low'])} to {_f(r['delta_high'])} | "
            f"{_f(r['p_greater'])} | {_f(r['p_holm'])} |"
        )
    return "\n".join(lines) + "\n"
