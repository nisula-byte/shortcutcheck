"""Addendum C (PREREGISTRATION-ADDENDUM-C.md): metamorphic augmentation fix, proof of concept.

Training and validation partitions only: this module never opens a locked partition or the
2015-16 holdout. The fix is fixed by the addendum and is not tuned here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.analysis.stats import holm, wilcoxon_greater
from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.evaluation.experiment import fit_mrs
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.mrs.base import MetamorphicRelation
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.runner.engine import run_mr, single_threaded
from shortcutcheck.runner.stats import summarise
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select

FILE = "PREREGISTRATION-ADDENDUM-C.md"
MLFLOW_EXPERIMENT = "addendum-c-fix-poc"
SPLITS = ("rq12-DR-AG", "rq12-DR-AG_deb", "rq12-VS-AG_deb", "rq3-AndroCT-2012-14-vs-2017-19")
ALGORITHMS = ("lr", "svm", "rf", "lgbm", "mlp")
SEEDS = (11, 22, 33)
EVAL_MRS = ("MR1", "MR3", "MR4", "MR5", "MR6")
HELD_OUT = ("MR1", "MR3")
META_ONLY = ("year", "apk_size")
FIXED_SEED = 11
RATE_RATIO = 0.5
F1_DROP = 0.02
ALPHA = 0.05
MIN_PASSING_GO = 3
ARMS = ("baseline", "augmented")
PAIR = ["split", "algorithm", "seed"]


# ---- data and the fix ---------------------------------------------------------------------


@dataclass
class SplitData:
    name: str
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    mrs: dict[str, MetamorphicRelation]
    realism: RealismCheck
    contrary: np.ndarray = field(repr=False)


def contrary_apps(
    x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series
) -> np.ndarray:
    """Validation apps a metadata-only LightGBM misclassifies (their metadata points wrong)."""
    cols = [c for c in META_ONLY if c in x_train.columns]
    model = build_model("lgbm", FIXED_SEED).fit(x_train[cols], y_train)
    with single_threaded(model):
        pred = (model.predict_proba(x_val[cols])[:, 1] >= 0.5).astype(int)
    return np.asarray(pred != y_val.to_numpy().astype(int), dtype=bool)


def load_split(cfg: Config, split: str) -> SplitData:
    """Training and validation partitions of one split, with MRs and realism fitted on training."""
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_tr, y_tr = select(load_part(cfg, split, "train"), "full", *args)
    x_va, y_va = select(load_part(cfg, split, "val"), "full", *args)
    x_va = x_va.reindex(columns=x_tr.columns)
    meta = [c for c in cfg.metadata_columns if c in x_tr.columns]
    realism = RealismCheck(seed=FIXED_SEED, metadata_columns=meta).fit(x_tr)
    mrs = {m.mr_id: m for m in fit_mrs(cfg, x_tr, y_tr, EVAL_MRS)}
    if "MR4" not in mrs:
        raise ValueError(f"{split}: MR4 does not apply, so the fix cannot be built")
    return SplitData(
        split, x_tr, y_tr, x_va, y_va, mrs, realism, contrary_apps(x_tr, y_tr, x_va, y_va)
    )


def augment(
    x: pd.DataFrame,
    y: pd.Series,
    mr4: MetamorphicRelation,
    realism: RealismCheck | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Training data plus one copy per app wearing the other class's metadata, label kept.

    The donor class is the other class relative to the app's *true* label (passed to MR4 as its
    ``predicted`` argument). Copies that fail MR4's validity check or the realism check are
    dropped; their number is returned.
    """
    labels = y.to_numpy().astype(int)
    copy = mr4.transform(x, rng=np.random.default_rng(seed), predicted=labels)
    ok = np.asarray(mr4.is_valid(copy), dtype=bool)
    if realism is not None and ok.any():
        real = np.zeros(len(copy), dtype=bool)
        real[ok] = realism.is_realistic(copy[ok])
        ok &= real
    x_aug = pd.concat([x, copy[ok]], ignore_index=True)
    y_aug = pd.concat([y, y[ok]], ignore_index=True)
    return x_aug, y_aug, int((~ok).sum())


# ---- training and evaluation --------------------------------------------------------------


def evaluate(model: Any, d: SplitData, seed: int) -> dict[str, float]:
    """Detection quality, contrary-app accuracy and every MR's violation rate on validation."""
    with single_threaded(model):
        scores = model.predict_proba(d.x_val)[:, 1]
        quality = detection_metrics(d.y_val, scores)
        correct = (scores >= 0.5).astype(int) == d.y_val.to_numpy().astype(int)
        out: dict[str, float] = {
            "f1": quality["f1"],
            "roc_auc": quality["roc_auc"],
            "contrary_acc": float(correct[d.contrary].mean()) if d.contrary.any() else np.nan,
            "n_contrary": float(d.contrary.sum()),
            "n_val": float(len(d.y_val)),
        }
        for mr_id in EVAL_MRS:
            if mr_id not in d.mrs:
                out[f"rate_{mr_id}"], out[f"n_{mr_id}"] = np.nan, 0.0
                continue
            s = summarise(run_mr(model, d.mrs[mr_id], d.x_val, seed=seed, realism=d.realism))
            out[f"rate_{mr_id}"], out[f"n_{mr_id}"] = s.rate, float(s.n_counted)
    return out


def run(
    cfg: Config,
    commit: str,
    splits: Sequence[str] = SPLITS,
    algorithms: Sequence[str] = ALGORITHMS,
    seeds: Sequence[int] = SEEDS,
    progress: Callable[[str], None] = lambda _m: None,
    log_mlflow: bool = True,
) -> pd.DataFrame:
    """Train both arms for every split, algorithm and seed; one row per model."""
    if log_mlflow:
        _setup_mlflow(cfg, MLFLOW_EXPERIMENT)
    rows = []
    for split in splits:
        d = load_split(cfg, split)
        for seed in seeds:
            x_aug, y_aug, rejected = augment(d.x_train, d.y_train, d.mrs["MR4"], d.realism, seed)
            for algorithm in algorithms:
                for arm in ARMS:
                    progress(f"{split} {algorithm} s{seed} {arm}")
                    x, y = (d.x_train, d.y_train) if arm == "baseline" else (x_aug, y_aug)
                    model = build_model(algorithm, seed).fit(x, y)  # type: ignore[arg-type]
                    measures = evaluate(model, d, seed)
                    row = {
                        "split": split,
                        "algorithm": algorithm,
                        "seed": seed,
                        "arm": arm,
                        "n_train": len(y),
                        "n_rejected_copies": rejected if arm == "augmented" else 0,
                        **measures,
                        "mlflow_run": "",
                    }
                    if log_mlflow:
                        with mlflow.start_run(run_name=f"{split}-{algorithm}-s{seed}-{arm}") as r:
                            mlflow.set_tags({"git_commit": commit, "addendum": "C"})
                            mlflow.log_params(
                                {
                                    k: row[k]
                                    for k in ("split", "algorithm", "seed", "arm", "n_train")
                                }
                            )
                            mlflow.log_metrics(
                                {k: float(v) for k, v in measures.items() if not pd.isna(v)}
                            )
                            row["mlflow_run"] = r.info.run_id
                    rows.append(row)
    return pd.DataFrame(rows)


# ---- decision rules -----------------------------------------------------------------------


def paired(rows: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Baseline and augmented values of one measure side by side, matched by algorithm and seed."""
    wide = rows.pivot_table(index=PAIR, columns="arm", values=measure, aggfunc="first")
    return wide.reindex(columns=list(ARMS)).reset_index()


def held_out_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """One-sided Wilcoxon (baseline > augmented) for MR1 and MR3 per split; Holm over all."""
    out = []
    for split in dict.fromkeys(rows["split"]):
        for mr in HELD_OUT:
            p = paired(rows[rows["split"] == split], f"rate_{mr}").dropna()
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
                    "p": wilcoxon_greater(d.tolist()).p_greater,
                }
            )
    frame = pd.DataFrame(
        out, columns=["split", "mr", "pairs", "mean_baseline", "mean_augmented", "p"]
    )
    frame["p_holm"] = holm(frame["p"].tolist()) if len(frame) else []
    return frame


@dataclass(frozen=True)
class SplitVerdict:
    split: str
    mr4_baseline: float
    mr4_augmented: float
    f1_baseline: float
    f1_augmented: float
    held_out_min_p_holm: float

    @property
    def a(self) -> bool:
        return bool(self.mr4_augmented <= RATE_RATIO * self.mr4_baseline)

    @property
    def b(self) -> bool:
        return bool(self.f1_baseline - self.f1_augmented <= F1_DROP)

    @property
    def c(self) -> bool:
        return bool(self.held_out_min_p_holm < ALPHA)

    @property
    def passes(self) -> bool:
        return self.a and self.b and self.c


@dataclass(frozen=True)
class CResult:
    splits: list[SplitVerdict]
    tests: pd.DataFrame

    @property
    def n_passing(self) -> int:
        return sum(v.passes for v in self.splits)

    @property
    def verdict(self) -> str:
        if self.n_passing >= MIN_PASSING_GO:
            return "GO"
        return "PARTIAL" if self.n_passing >= 1 else "NO-GO"


def judge(rows: pd.DataFrame) -> CResult:
    tests = held_out_tests(rows)
    verdicts = []
    for split in dict.fromkeys(rows["split"]):
        r = rows[rows["split"] == split]
        mr4, f1 = paired(r, "rate_MR4"), paired(r, "f1")
        t = tests[tests["split"] == split]
        verdicts.append(
            SplitVerdict(
                split,
                float(mr4["baseline"].mean()),
                float(mr4["augmented"].mean()),
                float(f1["baseline"].mean()),
                float(f1["augmented"].mean()),
                float(t["p_holm"].min()) if len(t) else float("nan"),
            )
        )
    return CResult(verdicts, tests)


# ---- report -------------------------------------------------------------------------------


def _f(v: Any) -> str:
    v = float(v)
    if pd.isna(v):
        return "n/a"
    return f"{v:.1e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}"


def _yes(b: bool) -> str:
    return "yes" if b else "no"


def secondary(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean per split and arm of every reported measure."""
    measures = ["f1", "roc_auc", "contrary_acc", *(f"rate_{m}" for m in EVAL_MRS)]
    means = rows.groupby(["split", "arm"], sort=False)[measures].mean().reset_index()
    return means


def render(rows: pd.DataFrame, result: CResult, head: str) -> str:
    lines = [
        "# Addendum C: metamorphic augmentation as a fix (proof of concept)",
        "",
        head,
        "",
        f"**Overall: {result.verdict}** ({result.n_passing} of {len(result.splits)} splits pass; "
        f"GO needs {MIN_PASSING_GO}).",
        "",
        "| Split | MR4 baseline → augmented | (a) ≤ 50% | F1 baseline → augmented "
        "| (b) drop ≤ 0.02 | (c) min Holm p, MR1/MR3 | (c) | Passes |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for v in result.splits:
        lines.append(
            f"| {v.split} | {_f(v.mr4_baseline)} → {_f(v.mr4_augmented)} | {_yes(v.a)} | "
            f"{_f(v.f1_baseline)} → {_f(v.f1_augmented)} | {_yes(v.b)} | "
            f"{_f(v.held_out_min_p_holm)} | {_yes(v.c)} | **{_yes(v.passes)}** |"
        )
    lines += [
        "",
        "## Held-out metadata MRs (rule c)",
        "",
        "One-sided Wilcoxon signed-rank, baseline > augmented, pairs matched by algorithm and "
        "seed; Holm over every row.",
        "",
        "| Split | MR | Pairs | Mean baseline | Mean augmented | p | p (Holm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in result.tests.to_dict("records"):
        lines.append(
            f"| {t['split']} | {t['mr']} | {t['pairs']} | {_f(t['mean_baseline'])} | "
            f"{_f(t['mean_augmented'])} | {_f(t['p'])} | {_f(t['p_holm'])} |"
        )
    lines += [
        "",
        "## Reported, not judged",
        "",
        "Means over 15 models per split and arm. MR5 and MR6 change permissions, not metadata "
        "(D20). Contrary-app accuracy: validation apps whose metadata alone points to the wrong "
        "class.",
        "",
    ]
    sec = secondary(rows)
    cols = [c for c in sec.columns if c not in ("split", "arm")]
    lines.append("| Split | Arm | " + " | ".join(cols) + " |")
    lines.append("| --- | --- |" + " ---: |" * len(cols))
    for vals in sec.to_dict("records"):
        lines.append(
            f"| {vals['split']} | {vals['arm']} | "
            + " | ".join(_f(float(vals[c])) for c in cols)
            + " |"
        )
    lines += ["", "Contrary-app accuracy, two-sided Wilcoxon (augmented vs baseline):", ""]
    for split in dict.fromkeys(rows["split"]):
        p = paired(rows[rows["split"] == split], "contrary_acc").dropna()
        if p.empty:
            continue
        d = (p["augmented"] - p["baseline"]).to_numpy()
        p_two = wilcoxon_greater((-d).tolist()).p_two_sided
        lines.append(f"- {split}: mean change {d.mean():+.3f}, p = {_f(p_two)}")
    rej = rows[rows["arm"] == "augmented"].groupby("split", sort=False)
    lines += ["", "Augmented copies rejected by the validity and realism checks (per seed):", ""]
    for name, g in rej:
        per_seed = g.drop_duplicates("seed")
        rejected = ", ".join(str(int(v)) for v in per_seed["n_rejected_copies"])
        sizes = ", ".join(str(int(v)) for v in per_seed["n_train"])
        lines.append(f"- {name!s}: rejected {rejected}; augmented training sizes {sizes}")
    lines += ["", "## Per algorithm (MR4 and F1, mean over seeds)", ""]
    means = rows.groupby(["split", "algorithm", "arm"], sort=False)[["rate_MR4", "f1"]].mean()
    per: pd.DataFrame = means.unstack("arm")  # type: ignore[assignment]
    lines.append(
        "| Split | Algorithm | MR4 baseline | MR4 augmented | F1 baseline | F1 augmented |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for key, r in per.iterrows():
        sp, alg = str(key[0]), str(key[1])  # type: ignore[index]
        lines.append(
            f"| {sp} | {alg} | {_f(r[('rate_MR4', 'baseline')])} | "
            f"{_f(r[('rate_MR4', 'augmented')])} | {_f(r[('f1', 'baseline')])} | "
            f"{_f(r[('f1', 'augmented')])} |"
        )
    return "\n".join(lines) + "\n"


def header(commit: str, prereg: str) -> str:
    return (
        f"Generated by `shortcutcheck addendum c` at commit `{commit}`. Rules: `{FILE}` "
        f"(committed in `{prereg[:7]}` before this experiment ran). Training and validation "
        "partitions only: no test data and not the 2015-16 holdout were read. It does not "
        "change H1-H3 or addenda A and B."
    )
