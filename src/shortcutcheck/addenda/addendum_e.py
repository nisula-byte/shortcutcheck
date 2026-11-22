"""Addendum E (PREREGISTRATION-ADDENDUM-E.md): rules and report for the AZL replication.

This module only judges saved tables. The data are read by ``evaluation.addendum_e_run``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shortcutcheck.addenda import addendum_d as d
from shortcutcheck.analysis.stats import holm, percentile_ci, wilcoxon_greater

FILE = "PREREGISTRATION-ADDENDUM-E.md"
MLFLOW_EXPERIMENT = "addendum-e-azl"
AZL_SPLIT = "rq3-AZL-2012-14-vs-2017-19"
AZL_HOLDOUT = "rq3-AZL-holdout-2015-16"
ARMS = ("baseline", "augmented", "duplicate", "behaviour")
SETTINGS = 6
SIGNALS = ("mr4_monitor", "psi", "confidence_drop", "atc_drop")
ALPHA = 0.05
RESAMPLES = 2000
SEED = 11


# ---- E3: algorithm-adjusted correlation -----------------------------------------------------


def adjusted_rho(signal: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
    """Pearson correlation of ranks after removing each group's mean rank (ties averaged)."""
    frame = pd.DataFrame({"s": signal, "t": target, "g": groups}).dropna()
    if len(frame) < 3:
        return float("nan")
    r = frame[["s", "t"]].rank()
    centred = r - r.groupby(frame["g"].to_numpy()).transform("mean")
    a, b = centred["s"].to_numpy(), centred["t"].to_numpy()
    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def adjusted_bootstrap(
    table: pd.DataFrame,
    signal: str,
    target: str = "decay",
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> np.ndarray:
    """Cluster bootstrap over configurations; ranks and centring redone in every resample."""
    rng = np.random.default_rng(seed)
    t = table.reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in t.groupby("configuration", sort=True)]
    s, y, alg = (t[signal].to_numpy(float), t[target].to_numpy(float), t["algorithm"].to_numpy())
    out = np.full(resamples, np.nan)
    for b in range(resamples):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out[b] = adjusted_rho(s[pick], y[pick], alg[pick])
    return out


@dataclass(frozen=True)
class Adjusted:
    subset: str
    signal: str
    rho: float
    low: float
    high: float
    n_models: int

    @property
    def holds(self) -> bool:
        return bool(self.rho > 0 and self.low > 0)


def judge_adjusted(
    table: pd.DataFrame, subset: str, signal: str = "mr4_monitor", resamples: int = RESAMPLES
) -> Adjusted:
    t = table.dropna(subset=[signal, "decay"])
    rho = adjusted_rho(
        t[signal].to_numpy(float), t["decay"].to_numpy(float), t["algorithm"].to_numpy()
    )
    low, high = percentile_ci(adjusted_bootstrap(t, signal, resamples=resamples))
    return Adjusted(subset, signal, rho, float(low), float(high), len(t))


# ---- E5: fix vs behaviour-only --------------------------------------------------------------


def judge_e5(rows: pd.DataFrame) -> pd.DataFrame:
    """Two-sided Wilcoxon, augmented vs behaviour-only, on holdout F1 and decay; Holm over 2."""
    out = []
    for measure in ("f1_holdout", "decay_1516"):
        p = d.paired(rows, measure, "augmented", "behaviour")
        diff = (p["augmented"] - p["behaviour"]).to_numpy()
        res = wilcoxon_greater(diff.tolist()) if len(diff) else None
        out.append(
            {
                "measure": measure,
                "pairs": len(diff),
                "mean_augmented": float(p["augmented"].mean()),
                "mean_behaviour": float(p["behaviour"].mean()),
                "mean_difference": float(diff.mean()) if len(diff) else float("nan"),
                "p_two_sided": res.p_two_sided if res else float("nan"),
            }
        )
    frame = pd.DataFrame(out)
    frame["p_holm"] = holm(frame["p_two_sided"].tolist())

    def verdict(r: pd.Series) -> str:
        if not r["p_holm"] < ALPHA:
            return "no difference"
        better_aug = (
            r["mean_difference"] > 0 if r["measure"] == "f1_holdout" else (r["mean_difference"] < 0)
        )
        return "fix better" if better_aug else "behaviour-only better"

    frame["verdict"] = frame.apply(verdict, axis=1)
    return frame


# ---- overall ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EResult:
    e1: d.H5aResult
    e2: d.H5bResult
    e3: Adjusted
    e5: pd.DataFrame

    @property
    def verdict(self) -> str:
        n = int(self.e1.holds) + int(self.e2.h5b)
        return {2: "REPLICATES", 1: "PARTIALLY REPLICATES"}.get(n, "DOES NOT REPLICATE")


def pool_table(pool: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the H5b judge expects (skew 0 for every model)."""
    t = pool.copy()
    t["skew"] = 0.0
    t["configuration"] = t["algorithm"].astype(str) + "|0.0|" + t["setting"].astype(str)
    return t


def judge(arms: pd.DataFrame, pool: pd.DataFrame, resamples: int = RESAMPLES) -> EResult:
    t = pool_table(pool)
    return EResult(
        d.judge_h5a(arms),
        d.judge_h5b(t, resamples=resamples),
        judge_adjusted(t, "AZL un-injected", resamples=resamples),
        judge_e5(arms),
    )


# ---- report ----------------------------------------------------------------------------------


def _f(v: object) -> str:
    return d._f(v)


def header(commit: str, prereg: str) -> str:
    return (
        f"Generated by `shortcutcheck addendum e` at commit `{commit}`. Rules: `{FILE}` "
        f"(committed in `{prereg[:7]}` before any data it tests was opened). It does not change "
        "H1-H5 or addenda A-D."
    )


def render(
    arms: pd.DataFrame,
    pool: pd.DataFrame,
    r: EResult,
    head: str,
    androct: Adjusted | None = None,
    androct_e5: pd.DataFrame | None = None,
) -> str:
    e17 = d.judge_h5a(arms, "decay_1719")
    lines = [
        "# Addendum E: replication on AZL (independent of AndroCT)",
        "",
        head,
        "",
        f"**Replication (E1 and E2): {r.verdict}.**",
        "",
        f"- **E1 (the fix reduces decay): {'HOLDS' if r.e1.holds else 'DOES NOT HOLD'}.** Mean "
        f"decay to 2015-16: baseline {_f(r.e1.decay_baseline)}, augmented "
        f"{_f(r.e1.decay_augmented)}; one-sided Wilcoxon over {r.e1.pairs} pairs p = "
        f"{_f(r.e1.p)}.",
        f"- **E2 (MR4 at monitoring time predicts decay): "
        f"{'HOLDS' if r.e2.h5b else 'DOES NOT HOLD'}.** rho = "
        f"{_f(r.e2.headline.rho['mr4_monitor'])} (95% CI "
        f"{_f(r.e2.headline.ci['mr4_monitor'][0])} to {_f(r.e2.headline.ci['mr4_monitor'][1])}),"
        f" {r.e2.headline.n_models} models.",
        f"- **E3 (holds within algorithms): {'HOLDS' if r.e3.holds else 'DOES NOT HOLD'}.** "
        f"Algorithm-adjusted rho = {_f(r.e3.rho)} (95% CI {_f(r.e3.low)} to {_f(r.e3.high)}).",
        "- **E4 (MR4 beats the monitors):** "
        + ", ".join(f"{k} {'holds' if v else 'does not hold'}" for k, v in r.e2.h5c().items())
        + ".",
        "- **E5 (fix vs behaviour-only, two-sided):** "
        + "; ".join(f"{x['measure']}: {x['verdict']}" for x in r.e5.to_dict("records"))
        + ".",
        "",
        "## E2 and E4: Spearman rho with 2015-16 decay (AZL pool)",
        "",
        "| Signal | rho | 95% CI | Algorithm-adjusted rho |",
        "| --- | ---: | --- | ---: |",
    ]
    t = pool_table(pool)
    for s in SIGNALS:
        adj = adjusted_rho(
            t[s].to_numpy(float), t["decay"].to_numpy(float), t["algorithm"].to_numpy()
        )
        lines.append(
            f"| {s} | {_f(r.e2.headline.rho[s])} | {_f(r.e2.headline.ci[s][0])} to "
            f"{_f(r.e2.headline.ci[s][1])} | {_f(adj)} |"
        )
    lines += [
        "",
        "| E4: MR4 monitor vs | delta rho | 95% CI | p | p (Holm) |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for c in r.e2.comparisons.to_dict("records"):
        lines.append(
            f"| {c['competitor']} | {_f(c['delta'])} | {_f(c['delta_low'])} to "
            f"{_f(c['delta_high'])} | {_f(c['p_greater'])} | {_f(c['p_holm'])} |"
        )
    lines += [
        "",
        "## E1 and E5: arms on AZL (means over 25 models)",
        "",
        *d.arm_means(
            arms,
            ["f1_val", "f1_holdout", "decay_1516", "f1_1719", "decay_1719", "rate_MR4_holdout"],
        ),
        "",
        f"- 2017-19 decay (reported): baseline {_f(e17.decay_baseline)}, augmented "
        f"{_f(e17.decay_augmented)}, one-sided p = {_f(e17.p)}.",
        "",
        "| E5 measure | Augmented | Behaviour-only | Mean difference | p (two-sided) | p (Holm) "
        "| Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for x in r.e5.to_dict("records"):
        lines.append(
            f"| {x['measure']} | {_f(x['mean_augmented'])} | {_f(x['mean_behaviour'])} | "
            f"{x['mean_difference']:+.3f} | {_f(x['p_two_sided'])} | {_f(x['p_holm'])} | "
            f"{x['verdict']} |"
        )
    lines += ["", "## Per algorithm: 2015-16 decay by arm (mean over seeds)", ""]
    per = arms.pivot_table(index="algorithm", columns="arm", values="decay_1516", aggfunc="mean")
    per = per.reindex(columns=[a for a in ARMS if a in per.columns])
    lines.append("| Algorithm | " + " | ".join(per.columns) + " |")
    lines.append("| --- |" + " ---: |" * len(per.columns))
    for alg, row in per.iterrows():
        lines.append(f"| {alg} | " + " | ".join(_f(v) for v in row) + " |")
    if androct is not None or androct_e5 is not None:
        lines += [
            "",
            "## Secondary, data seen before (not confirmatory): AndroCT 2015-16",
            "",
        ]
        if androct is not None:
            lines.append(
                f"- Algorithm-adjusted rho on the addendum D table ({androct.subset}, "
                f"{androct.n_models} models): {_f(androct.rho)} (95% CI {_f(androct.low)} to "
                f"{_f(androct.high)})."
            )
        if androct_e5 is not None:
            for x in androct_e5.to_dict("records"):
                lines.append(
                    f"- {x['measure']}: augmented {_f(x['mean_augmented'])}, behaviour-only "
                    f"{_f(x['mean_behaviour'])}, p (Holm) {_f(x['p_holm'])}: {x['verdict']}."
                )
    return "\n".join(lines) + "\n"
