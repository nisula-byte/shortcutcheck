"""H3 (RQ3, roadmap step 3.8), judged exactly as PREREGISTRATION.md states it.

Input: the pool table from steps 3.5-3.7, one row per model, with the columns in
``POOL_COLUMNS``. Every signal is oriented so that larger means "more decay expected".

* Headline subset: un-injected models (skew level 0): 6 settings x 5 algorithms = 30
  configurations, 150 models. The full pool is secondary.
* Cluster bootstrap: ``RESAMPLES`` resamples of configurations (algorithm x skew x setting),
  seed ``SEED``; all 5 seeds of a drawn configuration come with it. Percentile 95% CIs.
* **H3a**: rho(MR4 pre-deployment, F1 drop) >= ``RHO_MIN`` and its CI lower bound > 0.
* **H3b**: rho(MR4) - rho(SHAP metadata share) > 0. **H3c**: rho(MR4) - rho(b) > 0 for each
  monitoring baseline b (confidence drop, PSI, ATC). One-sided bootstrap p-values from the
  paired resamples; Holm over these 4 comparisons; a comparison holds if Holm p < ``ALPHA``.
* **Gate 3 outcome** (roadmap decision diagram): strong signal = H3a and H3b hold; signal,
  no edge = H3a holds, H3b does not; injected only = H3a fails on the headline subset but the
  same rule holds on the full pool; no signal = otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shortcutcheck.analysis.stats import (
    bootstrap_p_greater,
    cluster_bootstrap_rhos,
    holm,
    percentile_ci,
    spearman,
)

POOL_COLUMNS = (
    "model",  # unique model id
    "algorithm",
    "skew",  # injected skew level (0 = un-injected)
    "setting",  # hyperparameter setting index
    "seed",
    "mr4_pre",  # MR4 violation rate, 2012-14 validation inputs, realism filter on
    "shap_share",  # SHAP metadata share, 2012-14 validation inputs
    "confidence_drop",  # mean confidence on validation minus on 2017-19 inputs
    "psi",  # PSI of predicted probabilities, validation vs 2017-19 inputs
    "atc_drop",  # validation accuracy minus ATC-predicted 2017-19 accuracy
    "f1_drop",  # F1 on 2012-14 validation minus F1 on 2017-19 test
)
PRIMARY = "mr4_pre"
COMPETITORS = ("shap_share", "confidence_drop", "psi", "atc_drop")
H3B = "shap_share"
RHO_MIN = 0.5
ALPHA = 0.05
RESAMPLES = 2000
SEED = 11


def configuration(pool: pd.DataFrame) -> pd.Series:
    return (
        pool["algorithm"].astype(str)
        + "|"
        + pool["skew"].astype(str)
        + "|"
        + pool["setting"].astype(str)
    )


@dataclass(frozen=True)
class Correlations:
    subset: str
    n_models: int
    n_configurations: int
    rho: dict[str, float]
    ci: dict[str, tuple[float, float]]
    boot: pd.DataFrame  # resample x signal


def correlations(
    pool: pd.DataFrame,
    subset: str,
    signals: tuple[str, ...],
    target: str = "f1_drop",
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Correlations:
    t = pool.assign(_cluster=configuration(pool)).dropna(subset=[*signals, target])
    t = t.reset_index(drop=True)
    boot = cluster_bootstrap_rhos(t, signals, target, "_cluster", resamples, seed)
    return Correlations(
        subset,
        len(t),
        int(t["_cluster"].nunique()),
        {s: spearman(t[s], t[target]) for s in signals},
        {s: percentile_ci(boot[s].to_numpy()) for s in signals},
        boot,
    )


def rule_h3a(c: Correlations) -> bool:
    rho, low = c.rho[PRIMARY], c.ci[PRIMARY][0]
    return bool(rho >= RHO_MIN and low > 0)


@dataclass(frozen=True)
class H3Result:
    headline: Correlations
    full_pool: Correlations
    comparisons: pd.DataFrame  # one row per competitor: delta, CI, p, Holm p, holds
    missing: int  # models dropped for a missing value

    @property
    def h3a(self) -> bool:
        return rule_h3a(self.headline)

    @property
    def h3b(self) -> bool:
        return bool(self.comparisons.set_index("competitor").loc[H3B, "holds"])

    def h3c(self) -> dict[str, bool]:
        c = self.comparisons
        return {str(r.competitor): bool(r.holds) for r in c.itertuples() if r.competitor != H3B}

    @property
    def gate3(self) -> str:
        if self.h3a and self.h3b:
            return "Strong signal"
        if self.h3a:
            return "Signal, no edge"
        if rule_h3a(self.full_pool):
            return "Injected only"
        return "No signal"


def paired_comparisons(c: Correlations) -> pd.DataFrame:
    out = []
    for k in COMPETITORS:
        diff = (c.boot[PRIMARY] - c.boot[k]).to_numpy()
        low, high = percentile_ci(diff)
        out.append(
            {
                "competitor": k,
                "rho_mr4": c.rho[PRIMARY],
                "rho_competitor": c.rho[k],
                "delta": c.rho[PRIMARY] - c.rho[k],
                "delta_low": low,
                "delta_high": high,
                "p_greater": bootstrap_p_greater(diff),
                "n_valid_resamples": int((~np.isnan(diff)).sum()),
            }
        )
    table = pd.DataFrame(out)
    table["p_holm"] = holm(table["p_greater"].tolist())
    table["holds"] = table["p_holm"] < ALPHA
    return table


def judge_h3(pool: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED) -> H3Result:
    missing = [c for c in POOL_COLUMNS if c not in pool.columns]
    if missing:
        raise ValueError(f"Pool table lacks columns: {missing}")
    signals = (PRIMARY, *COMPETITORS)
    complete = pool.dropna(subset=[*signals, "f1_drop"])
    headline = correlations(
        complete[complete["skew"] == 0], "un-injected", signals, resamples=resamples, seed=seed
    )
    full = correlations(complete, "full pool", signals, resamples=resamples, seed=seed)
    return H3Result(headline, full, paired_comparisons(headline), len(pool) - len(complete))


# Secondary, descriptive only (no verdict): other signals, and decay per test year (D9).
SECONDARY_SIGNALS = ("mr1_pre", "mr3_pre", "core_mean_pre", "mr4_pre_unfiltered", "mr4_monitor")
SECONDARY_TARGETS = ("f1_drop_2017", "f1_drop_2018", "f1_drop_2019")


def secondary(pool: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED) -> pd.DataFrame:
    """rho and CI for every secondary signal/target present, on both subsets."""
    out = []
    for subset, part in (("un-injected", pool[pool["skew"] == 0]), ("full pool", pool)):
        pairs = [(s, "f1_drop") for s in SECONDARY_SIGNALS if s in pool.columns]
        pairs += [(PRIMARY, t) for t in SECONDARY_TARGETS if t in pool.columns]
        for signal, target in pairs:
            c = correlations(part, subset, (signal,), target, resamples, seed)
            out.append(
                {
                    "subset": subset,
                    "signal": signal,
                    "target": target,
                    "n_models": c.n_models,
                    "rho": c.rho[signal],
                    "low": c.ci[signal][0],
                    "high": c.ci[signal][1],
                }
            )
    return pd.DataFrame(out)


def _f(v: Any, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def render_h3(result: H3Result, header: str, extra: pd.DataFrame | None = None) -> str:
    h = result.headline
    lines = [
        "# RQ3: does the pre-deployment MR violation rate predict decay? (H3, step 3.8)",
        "",
        header,
        "",
        f"**Gate 3 outcome: {result.gate3}.** H3a {'holds' if result.h3a else 'does not hold'}; "
        f"H3b {'holds' if result.h3b else 'does not hold'}.",
        "",
        f"Headline subset: {h.subset}, {h.n_models} models in {h.n_configurations} "
        f"configurations. Models dropped for a missing value: {result.missing}.",
        "",
        "| Subset | Signal | Spearman rho with F1 drop | 95% cluster-bootstrap CI |",
        "| --- | --- | ---: | --- |",
    ]
    for c in (result.headline, result.full_pool):
        for s, rho in c.rho.items():
            lo, hi = c.ci[s]
            lines.append(f"| {c.subset} | {s} | {_f(rho)} | [{_f(lo)}, {_f(hi)}] |")
    lines += [
        "",
        f"H3a: rho(MR4) ≥ {RHO_MIN:g} and CI lower bound > 0 on the un-injected subset.",
        "",
        "## H3b and H3c: MR4 against each competitor (un-injected subset)",
        "",
        "| Competitor | rho(MR4) - rho(competitor) | 95% CI | p (one-sided) | p (Holm) | Holds |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for t in result.comparisons.itertuples():
        lines.append(
            f"| {t.competitor} | {_f(t.delta)} | [{_f(t.delta_low)}, {_f(t.delta_high)}] "
            f"| {_f(t.p_greater, 4)} | {_f(t.p_holm, 4)} | {'yes' if t.holds else 'no'} |"
        )
    if extra is not None and len(extra):
        lines += [
            "",
            "## Secondary (descriptive, no verdict)",
            "",
            "| Subset | Signal | Target | Models | rho | 95% CI |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
        for t in extra.itertuples():
            lines.append(
                f"| {t.subset} | {t.signal} | {t.target} | {t.n_models} | {_f(t.rho)} "
                f"| [{_f(t.low)}, {_f(t.high)}] |"
            )
    return "\n".join(lines) + "\n"
