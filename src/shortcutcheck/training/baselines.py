"""Step 1.8: baseline detectors and the shuffled-label leakage check.

Random forest and logistic regression, on full and behaviour-only features, over the five
configured seeds. Everything is evaluated on the *validation* partition; the test set stays
locked. A random forest trained on shuffled labels must score near chance on validation
(ROC-AUC near 0.5, PR-AUC near the malware prevalence). If it does not, something leaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from shortcutcheck.config import Config
from shortcutcheck.data.splits import load_part
from shortcutcheck.metrics import detection_metrics
from shortcutcheck.training.features import FeatureSet, select

Algorithm = Literal["rf", "lr", "svm", "lgbm", "mlp"]
ALGORITHMS: tuple[Algorithm, ...] = ("rf", "lr")  # step 1.8 baselines
# The proposal's five algorithms (pilot, RQ1-RQ3).
ALL_ALGORITHMS: tuple[Algorithm, ...] = ("lr", "svm", "rf", "lgbm", "mlp")
FEATURE_SETS: tuple[FeatureSet, ...] = ("full", "behaviour")


def build_model(
    algorithm: Algorithm, seed: int, n_estimators: int = 300, params: dict[str, Any] | None = None
) -> Pipeline:
    """Fixed, untuned settings: defaults plus class balancing (and scaling where needed).

    ``params`` overrides estimator settings for the RQ3 hyperparameter settings (configured in
    ``configs/experiments/rq3.yaml``). With no ``params`` the model is exactly the step 1.8 one.
    An unknown parameter name raises ``ValueError``.
    """
    pipe = _build_default(algorithm, seed, n_estimators)
    if params:
        prefix = "model__estimator__" if algorithm == "svm" else "model__"
        pipe.set_params(**{prefix + k: v for k, v in params.items()})
    return pipe


def _build_default(algorithm: str, seed: int, n_estimators: int) -> Pipeline:
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if algorithm == "svm":
        svm = LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)
        calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
        return Pipeline([impute, ("scale", StandardScaler()), ("model", calibrated)])
    if algorithm == "lgbm":
        lgbm = LGBMClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        return Pipeline([impute, ("model", lgbm)])
    if algorithm == "mlp":
        mlp = MLPClassifier(early_stopping=True, max_iter=300, random_state=seed)
        return Pipeline([impute, ("scale", StandardScaler()), ("model", mlp)])
    if algorithm == "rf":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def _align(train_x: pd.DataFrame, other_x: pd.DataFrame) -> pd.DataFrame:
    """Give validation the training columns, in training order (preprocessing fit on train only)."""
    return other_x.reindex(columns=train_x.columns)


@dataclass(frozen=True)
class ShuffledCheck:
    """Permutation test for pipeline leakage.

    Each permutation trains a model on shuffled training labels and scores it on validation.
    One permutation can land far from chance when a strong feature dominates the data, because
    the random function the model learns may line up with it. Only the *mean* over
    permutations is expected to be at chance, so that is what the check tests. A leak (for
    example, validation labels reaching training) pushes every permutation up together.
    """

    roc_aucs: tuple[float, ...]
    pr_aucs: tuple[float, ...]
    prevalence: float
    tolerance: float

    @staticmethod
    def _sem(values: tuple[float, ...]) -> float:
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    @property
    def roc_mean(self) -> float:
        return float(np.mean(self.roc_aucs))

    @property
    def pr_mean(self) -> float:
        return float(np.mean(self.pr_aucs))

    @property
    def roc_margin(self) -> float:
        return max(self.tolerance, 3 * self._sem(self.roc_aucs))

    @property
    def pr_margin(self) -> float:
        return max(self.tolerance, 3 * self._sem(self.pr_aucs))

    @property
    def passes(self) -> bool:
        """Judged on ROC-AUC only (decision D7).

        Swapping the class labels turns a model's ROC-AUC into 1 - ROC-AUC, so under the
        no-leak null its mean over permutations is 0.5. PR-AUC has no such symmetry: when one
        class sits in sparse regions of feature space, a shuffled-label model gives it extreme
        scores at *both* ends, which lifts that class's PR-AUC above its prevalence without any
        label information. PR-AUC is still reported, with the original criterion, but not judged.
        """
        return abs(self.roc_mean - 0.5) < self.roc_margin

    @property
    def pr_within_margin(self) -> bool:
        """The original (pre-D7) PR-AUC criterion, reported for transparency (rule 3)."""
        return abs(self.pr_mean - self.prevalence) < self.pr_margin


def _setup_mlflow(cfg: Config, experiment: str) -> None:
    uri = cfg.mlflow_tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri.removeprefix(prefix)).is_absolute():
        uri = prefix + (cfg.root / uri.removeprefix(prefix)).resolve().as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def run_baselines(
    cfg: Config, split_name: str, git_commit: str = "unknown"
) -> tuple[pd.DataFrame, ShuffledCheck]:
    train = load_part(cfg, split_name, "train")
    val = load_part(cfg, split_name, "val")
    _setup_mlflow(cfg, f"phase1-baselines-{split_name}")
    rows = []
    for algorithm in ALGORITHMS:
        for feature_set in FEATURE_SETS:
            train_x, train_y = select(
                train, feature_set, cfg.metadata_columns, cfg.id_column, cfg.dropped_features
            )
            val_x, val_y = select(
                val, feature_set, cfg.metadata_columns, cfg.id_column, cfg.dropped_features
            )
            val_x = _align(train_x, val_x)
            for seed in cfg.seeds:
                model = build_model(algorithm, seed).fit(train_x, train_y)
                scores = model.predict_proba(val_x)[:, 1]
                metrics = detection_metrics(val_y, scores)
                with mlflow.start_run(run_name=f"{algorithm}-{feature_set}-s{seed}") as run:
                    mlflow.set_tags({"step": "1.8", "split": split_name, "git_commit": git_commit})
                    mlflow.log_params(
                        {
                            "algorithm": algorithm,
                            "feature_set": feature_set,
                            "seed": seed,
                            "n_features": train_x.shape[1],
                            "eval_part": "val",
                        }
                    )
                    mlflow.log_metrics(metrics)
                    run_id = run.info.run_id
                rows.append(
                    {
                        "algorithm": algorithm,
                        "feature_set": feature_set,
                        "seed": seed,
                        **metrics,
                        "mlflow_run_id": run_id,
                    }
                )
    check = shuffled_label_checks(cfg, train, val, split_name, git_commit)
    return pd.DataFrame(rows), check


def shuffled_label_checks(
    cfg: Config,
    train: pd.DataFrame,
    val: pd.DataFrame,
    split_name: str,
    git_commit: str = "unknown",
) -> ShuffledCheck:
    train_x, train_y = select(
        train, "full", cfg.metadata_columns, cfg.id_column, cfg.dropped_features
    )
    val_x, val_y = select(val, "full", cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    val_x = _align(train_x, val_x)
    rocs, prs = [], []
    with mlflow.start_run(run_name="shuffled-label-permutation-test"):
        mlflow.set_tags({"step": "1.8", "split": split_name, "git_commit": git_commit})
        for i in range(cfg.shuffled_permutations):
            seed = cfg.split_seed * 1000 + i
            rng = np.random.default_rng(seed)
            shuffled = pd.Series(rng.permutation(train_y.to_numpy()), index=train_y.index)
            model = build_model("rf", seed, n_estimators=100).fit(train_x, shuffled)
            metrics = detection_metrics(val_y, model.predict_proba(val_x)[:, 1])
            rocs.append(metrics["roc_auc"])
            prs.append(metrics["pr_auc"])
            mlflow.log_metrics(
                {"perm_roc_auc": metrics["roc_auc"], "perm_pr_auc": metrics["pr_auc"]}, step=i
            )
        check = ShuffledCheck(
            roc_aucs=tuple(rocs),
            pr_aucs=tuple(prs),
            prevalence=float(val_y.mean()),
            tolerance=cfg.shuffled_label_tolerance,
        )
        mlflow.log_params({"algorithm": "rf", "labels": "shuffled", "permutations": len(rocs)})
        mlflow.log_metrics(
            {"roc_mean": check.roc_mean, "pr_mean": check.pr_mean, "passes": float(check.passes)}
        )
    return check


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds (rule 5)."""
    grouped = results.groupby(["algorithm", "feature_set"])
    metrics = ["f1", "pr_auc", "roc_auc", "fpr_at_95tpr"]
    mean = grouped[metrics].mean().add_suffix("_mean")
    sd = grouped[metrics].std(ddof=1).add_suffix("_sd")
    return mean.join(sd).reset_index()


def render_markdown(
    split_name: str, summary: pd.DataFrame, check: ShuffledCheck, git_commit: str
) -> str:
    lines = [
        f"# Phase 1 baselines: `{split_name}` (roadmap step 1.8)",
        "",
        f"Validation partition only; test set untouched. Git commit `{git_commit}`. "
        "Per-run MLflow IDs are in the CSV next to this file.",
        "",
        "| Algorithm | Features | F1 | PR-AUC | ROC-AUC | FPR@95%TPR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['algorithm']} | {r['feature_set']} | "
            + " | ".join(
                f"{r[m + '_mean']:.3f} ± {r[m + '_sd']:.3f}"
                for m in ("f1", "pr_auc", "roc_auc", "fpr_at_95tpr")
            )
            + " |"
        )
    c = check
    lines += [
        "",
        "## Shuffled-label check (permutation test)",
        "",
        f"{len(c.roc_aucs)} random forests trained on shuffled training labels, scored on "
        "validation. The check is judged on mean ROC-AUC, which must sit within the margin of "
        "0.5; margin = max(configured floor, 3 × standard error of the mean over "
        "permutations). PR-AUC is reported against the original criterion but not judged "
        "(decision D7: it is not centred on the prevalence when one class is more spread out "
        "in feature space).",
        "",
        "| Measure | Mean | Chance | Margin | Range over permutations | Within margin | Judged |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| ROC-AUC | {c.roc_mean:.3f} | 0.500 | {c.roc_margin:.3f} "
        f"| {min(c.roc_aucs):.3f}–{max(c.roc_aucs):.3f} "
        f"| {'yes' if c.passes else 'no'} | yes |",
        f"| PR-AUC | {c.pr_mean:.3f} | {c.prevalence:.3f} | {c.pr_margin:.3f} "
        f"| {min(c.pr_aucs):.3f}–{max(c.pr_aucs):.3f} "
        f"| {'yes' if c.pr_within_margin else 'no'} | no |",
        "",
        f"**Check: {'PASS' if c.passes else 'FAIL'}.** "
        + (
            "Shuffled-label models score at chance on average."
            if c.passes
            else "Shuffled-label models score away from chance on average: suspect leakage. "
            "Stop and investigate before Gate 1."
        ),
        "",
    ]
    return "\n".join(lines)
