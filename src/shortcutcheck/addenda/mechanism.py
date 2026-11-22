"""Addendum F (PREREGISTRATION-ADDENDUM-F.md): why the fix behaves like dropping metadata.

Training and validation partitions only: this module never opens a locked partition or a
holdout. The fix is addendum C's, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shortcutcheck.addenda import addendum_d as d
from shortcutcheck.addenda.fix_poc import augment
from shortcutcheck.addenda.phase_a import cv_auc
from shortcutcheck.analysis.stats import holm, wilcoxon_greater
from shortcutcheck.config import Config
from shortcutcheck.data.dada import split_feature_groups
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.addendum_d_run import train_data
from shortcutcheck.runner.engine import malware_probability
from shortcutcheck.training.baselines import build_model
from shortcutcheck.training.features import select

FILE = "PREREGISTRATION-ADDENDUM-F.md"
SPLITS = (
    "rq12-DR-AG",
    "rq12-DR-AG_deb",
    "rq3-AndroCT-2012-14-vs-2017-19",
    "rq3-AZL-2012-14-vs-2017-19",
)
ALGORITHMS = ("lr", "svm", "rf", "lgbm", "mlp")
LINEAR = ("lr", "svm")
TREES = ("rf", "lgbm")
SEEDS = (11, 22, 33, 44, 55)
ARMS = ("baseline", "augmented", "behaviour")
PERMUTATIONS = 5
SEED = 11
SEPARABILITY_MAX = 0.60
HALVING = 0.5
MIN_R = 0.005
MIN_SPLITS = 3
ALPHA = 0.05


def reliance(model: Any, x: pd.DataFrame, y: pd.Series, metadata: list[str]) -> tuple[float, float]:
    """(validation ROC-AUC, drop in ROC-AUC when the metadata columns are permuted jointly)."""
    yy = y.to_numpy().astype(int)
    auc = float(roc_auc_score(yy, malware_probability(model, x)))
    rng = np.random.default_rng(SEED)
    drops = []
    for _ in range(PERMUTATIONS):
        xp = x.copy()
        xp[metadata] = x[metadata].to_numpy()[rng.permutation(len(x))]
        drops.append(auc - float(roc_auc_score(yy, malware_probability(model, xp))))
    return auc, float(np.mean(drops))


def agreement(model: Any, reference: Any, x: pd.DataFrame) -> float:
    a = malware_probability(model, x) >= 0.5
    b = malware_probability(reference, x) >= 0.5
    return float((a == b).mean())


def run(
    cfg: Config,
    splits: Sequence[str] = SPLITS,
    algorithms: Sequence[str] = ALGORITHMS,
    seeds: Sequence[int] = SEEDS,
    progress: Callable[[str], None] = lambda _m: None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per split, algorithm, seed and arm; and one separability row per split."""
    rows, sep = [], []
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    for split in splits:
        tr = train_data(cfg, split)
        metadata = [c for c in cfg.metadata_columns if c in tr.x.columns]
        _, behaviour = split_feature_groups(list(tr.x.columns), metadata)
        x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
        x_val = x_val.reindex(columns=tr.x.columns)
        x_aug11, y_aug11, _ = augment(tr.x, tr.y, tr.mrs["MR4"], tr.realism, SEED)
        sep.append(
            {
                "split": split,
                "meta_auc_original": cv_auc(tr.x[metadata], tr.y, "lgbm"),
                "meta_auc_augmented": cv_auc(
                    x_aug11[metadata].reset_index(drop=True), y_aug11.reset_index(drop=True), "lgbm"
                ),
            }
        )
        progress(f"{split} separability done")
        for seed in seeds:
            x_aug, y_aug, _ = augment(tr.x, tr.y, tr.mrs["MR4"], tr.realism, seed)
            data = {
                "baseline": (tr.x, tr.y),
                "augmented": (x_aug, y_aug),
                "behaviour": (tr.x[behaviour], tr.y),
            }
            for algorithm in algorithms:
                models = {
                    a: build_model(algorithm, seed).fit(*data[a])  # type: ignore[arg-type]
                    for a in ARMS
                }
                for a in ARMS:
                    auc, r = reliance(models[a], x_val, y_val, metadata)
                    rows.append(
                        {
                            "split": split,
                            "algorithm": algorithm,
                            "seed": seed,
                            "arm": a,
                            "val_auc": auc,
                            "reliance": r,
                            "agreement_behaviour": agreement(models[a], models["behaviour"], x_val),
                        }
                    )
                progress(f"{split} {algorithm} s{seed}")
    return pd.DataFrame(rows), pd.DataFrame(sep)


# ---- rules -----------------------------------------------------------------------------------


def _mean(rows: pd.DataFrame, split: str, algs: Sequence[str], arm: str, col: str) -> float:
    r = rows[(rows["split"] == split) & rows["algorithm"].isin(algs) & (rows["arm"] == arm)]
    return float(r[col].mean())


@dataclass(frozen=True)
class FResult:
    f1: bool
    f2: str
    f3: str
    f4: bool
    per_split: pd.DataFrame
    f4_tests: pd.DataFrame

    @property
    def supported(self) -> bool:
        return self.f1 and self.f2 == "HOLDS" and self.f4


def judge(rows: pd.DataFrame, sep: pd.DataFrame) -> FResult:
    f1 = bool((sep["meta_auc_augmented"] <= SEPARABILITY_MAX).all())
    out = []
    for split in dict.fromkeys(rows["split"]):
        rec: dict[str, Any] = {"split": split}
        for alg in (*LINEAR, *TREES, "mlp"):
            rec[f"R_base_{alg}"] = _mean(rows, split, (alg,), "baseline", "reliance")
            rec[f"R_aug_{alg}"] = _mean(rows, split, (alg,), "augmented", "reliance")
        lin_base = np.mean([rec[f"R_base_{a}"] for a in LINEAR])
        tree_base = np.mean([rec[f"R_base_{a}"] for a in TREES])
        rec["informative_f2"] = bool(lin_base >= MIN_R)
        rec["informative_f3"] = bool(lin_base >= MIN_R and tree_base >= MIN_R)
        rec["f2_split"] = all(rec[f"R_aug_{a}"] <= HALVING * rec[f"R_base_{a}"] for a in LINEAR)
        red = {
            a: 1 - rec[f"R_aug_{a}"] / rec[f"R_base_{a}"]
            for a in (*LINEAR, *TREES)
            if rec[f"R_base_{a}"] > 0
        }
        rec["reduction_linear"] = (
            float(np.mean([red[a] for a in LINEAR if a in red])) if red else np.nan
        )
        rec["reduction_trees"] = (
            float(np.mean([red[a] for a in TREES if a in red])) if red else np.nan
        )
        rec["f3_split"] = bool(rec["reduction_trees"] < rec["reduction_linear"])
        out.append(rec)
    per = pd.DataFrame(out)

    def count(flag: str, informative: str) -> str:
        inf = per[per[informative]]
        if len(inf) < MIN_SPLITS:
            return "NOT TESTABLE"
        return "HOLDS" if int(inf[flag].sum()) >= MIN_SPLITS else "DOES NOT HOLD"

    tests = []
    for split in per["split"]:
        r = rows[(rows["split"] == split) & rows["algorithm"].isin(LINEAR)]
        p = d.paired(r.assign(split=split), "agreement_behaviour", "baseline", "augmented")
        diff = (p["augmented"] - p["baseline"]).to_numpy()
        tests.append(
            {
                "split": split,
                "pairs": len(diff),
                "agreement_baseline": float(p["baseline"].mean()),
                "agreement_augmented": float(p["augmented"].mean()),
                "p": wilcoxon_greater(diff.tolist()).p_greater if len(diff) else np.nan,
            }
        )
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t["p"].tolist())
    f4 = bool(int((t["p_holm"] < ALPHA).sum()) >= MIN_SPLITS)
    return FResult(
        f1, count("f2_split", "informative_f2"), count("f3_split", "informative_f3"), f4, per, t
    )


def render(rows: pd.DataFrame, sep: pd.DataFrame, r: FResult, head: str) -> str:
    f = d._f
    lines = [
        "# Addendum F: why the fix behaves like dropping metadata",
        "",
        head,
        "",
        f"**Mechanism {'SUPPORTED' if r.supported else 'NOT SUPPORTED'}** (needs F1, F2 and F4).",
        "",
        "- **F1 (metadata alone stop separating the classes): "
        f"{'HOLDS' if r.f1 else 'DOES NOT HOLD'}.**",
        f"- **F2 (linear models stop relying on metadata): {r.f2}.**",
        f"- **F3 (tree ensembles retain more reliance): {r.f3}.**",
        f"- **F4 (linear models converge to the metadata-free model): "
        f"{'HOLDS' if r.f4 else 'DOES NOT HOLD'}.**",
        "",
        "## F1: metadata-only cross-validated ROC-AUC (LightGBM) on the training set",
        "",
        "| Split | Original | Augmented |",
        "| --- | ---: | ---: |",
    ]
    for s in sep.to_dict("records"):
        lines.append(
            f"| {s['split']} | {f(s['meta_auc_original'])} | {f(s['meta_auc_augmented'])} |"
        )
    lines += [
        "",
        "## F2 and F3: metadata reliance R (drop in validation ROC-AUC when metadata are permuted)",
        "",
        "| Split | Algorithm | R baseline | R augmented | R behaviour-only |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in dict.fromkeys(rows["split"]):
        for alg in ALGORITHMS:
            vals = [_mean(rows, split, (alg,), a, "reliance") for a in ARMS]
            lines.append(f"| {split} | {alg} | " + " | ".join(f(v) for v in vals) + " |")
    lines += [
        "",
        "| Split | Informative (F2/F3) | F2 holds here | Reduction, linear | Reduction, trees "
        "| F3 holds here |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for p in r.per_split.to_dict("records"):
        lines.append(
            f"| {p['split']} | {p['informative_f2']}/{p['informative_f3']} | {p['f2_split']} | "
            f"{f(p['reduction_linear'])} | {f(p['reduction_trees'])} | {p['f3_split']} |"
        )
    lines += [
        "",
        "## F4: agreement with the behaviour-only model (LR and SVM, 10 pairs per split)",
        "",
        "| Split | Baseline | Augmented | p | p (Holm) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for t in r.f4_tests.to_dict("records"):
        lines.append(
            f"| {t['split']} | {f(t['agreement_baseline'])} | {f(t['agreement_augmented'])} | "
            f"{f(t['p'])} | {f(t['p_holm'])} |"
        )
    lines += ["", "## Validation ROC-AUC by arm (reported)", ""]
    lines += d.arm_means(rows, ["val_auc", "reliance", "agreement_behaviour"])
    return "\n".join(lines) + "\n"


def header(commit: str, prereg: str) -> str:
    return (
        f"Generated by `shortcutcheck addendum f` at commit `{commit}`. Rules: `{FILE}` "
        f"(committed in `{prereg[:7]}` before anything below was computed). Training and "
        "validation partitions only; no test data and no holdout were read."
    )
