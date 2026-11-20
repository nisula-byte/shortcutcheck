"""Detection metrics. Accuracy is deliberately absent: classes are imbalanced."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def fpr_at_tpr(y_true: ArrayLike, score: ArrayLike, target_tpr: float = 0.95) -> float:
    """Lowest false-positive rate at which the true-positive rate reaches ``target_tpr``."""
    fpr, tpr, _ = roc_curve(y_true, score)
    reached = np.flatnonzero(tpr >= target_tpr)
    return float(fpr[reached[0]]) if reached.size else 1.0


def detection_metrics(
    y_true: ArrayLike, score: ArrayLike, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    return {
        "f1": float(f1_score(y, (s >= threshold).astype(int), zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "fpr_at_95tpr": fpr_at_tpr(y, s, 0.95),
        "prevalence": float(y.mean()),
    }
