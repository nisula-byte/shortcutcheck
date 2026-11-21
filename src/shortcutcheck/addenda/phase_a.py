"""Addendum A (PREREGISTRATION-ADDENDUM-A.md): A1 mechanism, A2 validity, A3 plain Gower."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shortcutcheck.analysis.h3 import correlations
from shortcutcheck.analysis.stats import cluster_bootstrap_rhos, percentile_ci, spearman
from shortcutcheck.config import Config
from shortcutcheck.data.dada import LABEL
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.experiment import fit_mrs, stratified_subsample
from shortcutcheck.experiments.spec import Experiment
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.runner.engine import run_mr
from shortcutcheck.training.baselines import build_model
from shortcutcheck.training.features import select
from shortcutcheck.training.pool import pool_members

FILE = "PREREGISTRATION-ADDENDUM-A.md"
SEED = 11
SEEDS = (11, 22, 33, 44, 55)
FAMILIES = {"DR": ("rq12-DR-AG", "rq12-DR-AG_deb"), "VS": ("rq12-VS-AG", "rq12-VS-AG_deb")}
RAW_META = ["year", "apk_size"]
BINNED = ["Year", "APK size"]
MARGIN = 0.02
PROXY_MAX = 0.60
RHO_MIN = 0.5
RESAMPLES = 2000

# ---- A1: mechanism ---------------------------------------------------------------------------


def cv_auc(x: pd.DataFrame, y: pd.Series, algorithm: str, seed: int = SEED) -> float:
    """Mean ROC-AUC of 5-fold stratified cross-validation of ``build_model(algorithm, 11)``."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train, test in folds.split(x, y):
        model = build_model(algorithm, SEED).fit(x.iloc[train], y.iloc[train])  # type: ignore[arg-type]
        p = model.predict_proba(x.iloc[test])[:, 1]
        scores.append(roc_auc_score(y.iloc[test], p))
    return float(np.mean(scores))


def groups(cfg: Config, frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    x, y = select(frame, "full", cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    behaviour = [c for c in x.columns if c not in cfg.metadata_columns]
    return {
        "M1 raw metadata": x[RAW_META],
        "M2 binned proxies": frame[BINNED].apply(pd.to_numeric, errors="coerce"),
        "M3 behaviour": x[behaviour],
    }, y


def a1_measures(cfg: Config, algorithms: tuple[str, ...] = ("lgbm", "lr")) -> pd.DataFrame:
    rows = []
    for family, (biased, debiased) in FAMILIES.items():
        deb_frame = load_part(cfg, debiased, "train")
        deb_groups, deb_y = groups(cfg, deb_frame)
        big_frame = load_part(cfg, biased, "train")
        for algorithm in algorithms:
            for group, x in deb_groups.items():
                rows.append(
                    {
                        "family": family,
                        "set": debiased,
                        "arm": "debiased",
                        "algorithm": algorithm,
                        "group": group,
                        "auc": cv_auc(x, deb_y, algorithm),
                        "sd": 0.0,
                    }
                )
            scores: dict[str, list[float]] = {g: [] for g in deb_groups}
            for seed in SEEDS:
                keep = stratified_subsample(big_frame[LABEL], len(deb_frame), seed)
                sub_groups, sub_y = groups(cfg, big_frame.iloc[keep].reset_index(drop=True))
                for group, x in sub_groups.items():
                    scores[group].append(cv_auc(x, sub_y, algorithm))
            for group, vals in scores.items():
                rows.append(
                    {
                        "family": family,
                        "set": biased,
                        "arm": "biased, size-matched",
                        "algorithm": algorithm,
                        "group": group,
                        "auc": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class A1Verdict:
    family: str
    a_holds: bool
    b_holds: bool
    values: dict[str, float]


def judge_a1(measures: pd.DataFrame) -> list[A1Verdict]:
    m = measures[measures["algorithm"] == "lgbm"]
    out = []
    for family in FAMILIES:
        f = m[m["family"] == family].set_index(["arm", "group"])["auc"]
        v = {
            "M1_deb": f[("debiased", "M1 raw metadata")],
            "M1_bias": f[("biased, size-matched", "M1 raw metadata")],
            "M2_deb": f[("debiased", "M2 binned proxies")],
            "M3_deb": f[("debiased", "M3 behaviour")],
            "M3_bias": f[("biased, size-matched", "M3 behaviour")],
        }
        a = v["M2_deb"] <= PROXY_MAX and v["M1_deb"] >= v["M1_bias"] - MARGIN
        b = v["M3_deb"] <= v["M3_bias"] - MARGIN
        out.append(A1Verdict(family, bool(a), bool(b), {k: float(x) for k, x in v.items()}))
    return out


# ---- A2: convergent validity ----------------------------------------------------------------


def a2_table(
    rows: pd.DataFrame, detection: pd.DataFrame, mr: str = "MR4", metric: str = "f1"
) -> pd.DataFrame:
    r = rows[(rows["stratum"] == "all") & (rows["mr"] == mr) & (rows["feature_set"] == "full")]
    viol = r[["train_split", "algorithm", "seed", "rate"]]
    d = detection.pivot_table(
        index=["train_split", "algorithm", "seed"], columns="feature_set", values=metric
    ).reset_index()
    d["gain"] = d["full"] - d["behaviour"]
    t = viol.merge(
        d[["train_split", "algorithm", "seed", "gain"]], on=["train_split", "algorithm", "seed"]
    )
    t["cluster"] = t["train_split"] + "|" + t["algorithm"]
    return t.reset_index(drop=True)


@dataclass(frozen=True)
class A2Result:
    n: int
    clusters: int
    rho: float
    ci: tuple[float, float]

    @property
    def holds(self) -> bool:
        return bool(self.rho >= RHO_MIN and self.ci[0] > 0)


def judge_a2(table: pd.DataFrame, resamples: int = RESAMPLES) -> A2Result:
    boot = cluster_bootstrap_rhos(table, ["rate"], "gain", "cluster", resamples, SEED)
    return A2Result(
        len(table),
        int(table["cluster"].nunique()),
        spearman(table["rate"], table["gain"]),
        percentile_ci(boot["rate"].to_numpy()),
    )


# ---- A3: plain-Gower sensitivity --------------------------------------------------------------


def a3_rates(cfg: Config, exp: Experiment, model_dir: Path) -> pd.DataFrame:
    """Pre-deployment MR4 rate with plain-Gower realism, every pool model (validation only)."""
    split = str(exp.pool["split"])
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_train, y_train = select(load_part(cfg, split, "train"), "full", *args)
    x_val, _ = select(load_part(cfg, split, "val"), "full", *args)
    x_val = x_val.reindex(columns=x_train.columns)
    (mr4,) = fit_mrs(cfg, x_train, y_train, ("MR4",))
    plain = RealismCheck(seed=exp.seeds[0], metadata_columns=None).fit(x_train)
    rows: list[dict[str, Any]] = []
    for m in pool_members(exp):
        model = joblib.load(model_dir / f"{m.model}.joblib")
        res = run_mr(model, mr4, x_val, seed=m.seed, realism=plain)
        rows.append(
            {
                "model": m.model,
                "mr4_pre_plain": res.violation_rate,
                "n_unrealistic_plain": int(res.n_unrealistic),
            }
        )
    return pd.DataFrame(rows)


def a3_correlations(
    pool: pd.DataFrame, plain: pd.DataFrame, resamples: int = RESAMPLES
) -> pd.DataFrame:
    t = pool.merge(plain, on="model", validate="one_to_one")
    out = []
    for subset, part in (("un-injected", t[t["skew"] == 0]), ("full pool", t)):
        c = correlations(part, subset, ("mr4_pre", "mr4_pre_plain"), resamples=resamples)
        for s in ("mr4_pre", "mr4_pre_plain"):
            out.append(
                {
                    "subset": subset,
                    "signal": s,
                    "n_models": c.n_models,
                    "rho": c.rho[s],
                    "low": c.ci[s][0],
                    "high": c.ci[s][1],
                }
            )
    return pd.DataFrame(out)


# ---- reports ---------------------------------------------------------------------------------


def _f(v: Any, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def header(step: str, commit: str, prereg: str) -> str:
    return (
        f"Generated by `shortcutcheck addendum {step}` at commit `{commit}`. Rules: "
        f"`{FILE}` (committed in `{prereg[:7]}` before this analysis ran). Explanatory analysis "
        "with pre-stated rules; it does not change the H1-H3 verdicts."
    )


def render_a1(measures: pd.DataFrame, verdicts: list[A1Verdict], head: str) -> str:
    lines = ["# A1: why debiased training increased metadata reliance (H2 mechanism)", "", head, ""]
    for v in verdicts:
        lines.append(
            f"- **{v.family} family:** (a) proxy debiasing leaves the raw shortcut: "
            f"**{'holds' if v.a_holds else 'does not hold'}**; (b) weaker behaviour signal: "
            f"**{'holds' if v.b_holds else 'does not hold'}**."
        )
    lines += [
        "",
        "Cross-validated ROC-AUC of a model trained on one feature group only "
        "(biased sets: mean ± SD over 5 size-matched subsamples).",
        "",
        "| Family | Set | Algorithm | M1 raw metadata | M2 binned proxies | M3 behaviour |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (fam, arm, alg), g in measures.groupby(["family", "arm", "algorithm"], sort=False):
        cells = []
        for grp in ("M1 raw metadata", "M2 binned proxies", "M3 behaviour"):
            r = g[g["group"] == grp].iloc[0]
            cells.append(_f(r["auc"]) + (f" ± {_f(r['sd'])}" if r["sd"] else ""))
        lines.append(f"| {fam} | {arm} | {alg} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_a2(results: dict[str, A2Result], head: str) -> str:
    main = results["MR4 vs F1 gain"]
    lines = [
        "# A2: does the violation rate measure real metadata reliance?",
        "",
        head,
        "",
        f"**Convergent validity {'holds' if main.holds else 'does not hold'}** "
        f"(rule: ρ ≥ {RHO_MIN} and CI lower bound > 0).",
        "",
        "| Pairing | Models | Configurations | Spearman ρ | 95% cluster-bootstrap CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.n} | {r.clusters} | {_f(r.rho)} | " f"[{_f(r.ci[0])}, {_f(r.ci[1])}] |"
        )
    lines += ["", "Only the first row has a verdict; the others are reported without one."]
    return "\n".join(lines) + "\n"


def render_a3(corr: pd.DataFrame, plain: pd.DataFrame, head: str) -> str:
    lines = [
        "# A3: plain-Gower realism sensitivity for RQ3",
        "",
        head,
        "",
        f"{len(plain)} pool models; mean MR4 rate with plain Gower "
        f"{plain['mr4_pre_plain'].mean():.3f}. No verdict (secondary analysis).",
        "",
        "| Subset | Signal | Models | ρ with F1 drop | 95% cluster-bootstrap CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for t in corr.itertuples():
        lines.append(
            f"| {t.subset} | {t.signal} | {t.n_models} | {_f(t.rho)} | "
            f"[{_f(t.low)}, {_f(t.high)}] |"
        )
    return "\n".join(lines) + "\n"
