"""Label-free RQ3 signals (roadmap step 3.6), defined in PREREGISTRATION.md section 5.

Every function here works on model outputs or feature frames. None of them takes the label of
an input it scores: frames are checked by ``assert_unlabelled`` and a test inspects every
signature (roadmap 3.6 Check). The one exception is the ATC threshold, which the
pre-registration fits on *validation* labels (``atc_threshold``); the 2017-19 side of ATC
(``atc_predicted_accuracy``) takes probabilities only.

Signals (all oriented so that larger = more decay expected):

* ``confidence_drop``: mean max(p, 1-p) on validation minus the same on test inputs.
* ``psi``: population stability index of predicted malware probabilities, validation vs test;
  10 bins with edges at the validation deciles (duplicate edges merged), shares floored at 1e-4.
* ``atc_drop``: validation accuracy minus the ATC-predicted test accuracy.
* ``shap_metadata_share``: mean over inputs of sum|SHAP| on metadata columns / sum|SHAP| on all.
* MR violation rates come from ``runner.engine.run_mr`` (label-free by construction).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from shortcutcheck.data.dada import LABEL
from shortcutcheck.runner.engine import single_threaded

LABEL_LIKE = (LABEL, "label", "labels", "y", "class", "dataset")


class LabelLeakError(ValueError):
    """A label column reached a label-free signal."""


def assert_unlabelled(x: pd.DataFrame) -> pd.DataFrame:
    leaked = [c for c in x.columns if str(c).lower() in LABEL_LIKE]
    if leaked:
        raise LabelLeakError(f"Label-free signal received label column(s): {leaked}")
    return x


def confidence(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.asarray(np.maximum(p, 1 - p), dtype=float)


def confidence_drop(p_val: np.ndarray, p_test: np.ndarray) -> float:
    return float(confidence(p_val).mean() - confidence(p_test).mean())


def psi_edges(p_val: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inner bin edges at the validation deciles, duplicates merged."""
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(p_val, dtype=float), qs))


def psi(p_val: np.ndarray, p_test: np.ndarray, bins: int = 10, floor: float = 1e-4) -> float:
    edges = psi_edges(p_val, bins)

    def shares(p: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(edges, np.asarray(p, dtype=float), side="right")
        counts = np.bincount(idx, minlength=len(edges) + 1)
        return np.maximum(counts / max(1, len(p)), floor)

    v, t = shares(p_val), shares(p_test)
    return float(np.sum((t - v) * np.log(t / v)))


def atc_threshold(conf_val: np.ndarray, correct_val: np.ndarray) -> float:
    """t such that the share of validation inputs with confidence > t equals val accuracy."""
    acc = float(np.mean(correct_val))
    return float(np.quantile(np.asarray(conf_val, dtype=float), 1 - acc))


def atc_predicted_accuracy(threshold: float, conf_test: np.ndarray) -> float:
    return float(np.mean(np.asarray(conf_test, dtype=float) > threshold))


def atc_drop(val_accuracy: float, threshold: float, conf_test: np.ndarray) -> float:
    return float(val_accuracy - atc_predicted_accuracy(threshold, conf_test))


# ---- SHAP metadata share ---------------------------------------------------------------------

EXPLAINERS = {
    "rf": "tree",
    "lgbm": "tree",
    "lr": "linear",
    "svm": "permutation",
    "mlp": "permutation",
}


def _positive_class(values: Any) -> np.ndarray:
    v = values.values if hasattr(values, "values") else values
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v, dtype=float)
    return v[:, :, 1] if v.ndim == 3 else v


def shap_values(
    model: Any, algorithm: str, x_explain: pd.DataFrame, x_background: pd.DataFrame, seed: int
) -> np.ndarray:
    """SHAP values (inputs x features) for the malware output, per the pre-registered explainer."""
    import shap

    assert_unlabelled(x_explain)
    assert_unlabelled(x_background)
    kind = EXPLAINERS[algorithm]
    cols = list(model.feature_names_in_)
    xe, xb = x_explain[cols], x_background[cols]
    if kind in ("tree", "linear"):
        prep, est = model[:-1], model[-1]
        te = np.asarray(prep.transform(xe), dtype=float)
        tb = np.asarray(prep.transform(xb), dtype=float)
        if kind == "tree":
            explainer = shap.TreeExplainer(
                est, data=tb, model_output="probability", feature_perturbation="interventional"
            )
            with single_threaded(model):
                return _positive_class(explainer.shap_values(te, check_additivity=False))
        explainer = shap.LinearExplainer(est, shap.maskers.Independent(tb, max_samples=len(tb)))
        return _positive_class(explainer.shap_values(te))

    def f(x: np.ndarray) -> np.ndarray:
        with single_threaded(model):
            return np.asarray(model.predict_proba(pd.DataFrame(x, columns=cols))[:, 1])

    masker = shap.maskers.Independent(xb.to_numpy(dtype=float), max_samples=len(xb))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    return _positive_class(explainer(xe.to_numpy(dtype=float), silent=True))


def metadata_share(values: np.ndarray, columns: Sequence[str], metadata: Sequence[str]) -> float:
    """Mean over inputs of sum|phi| on metadata columns / sum|phi| on all columns."""
    a = np.abs(np.asarray(values, dtype=float))
    meta = np.isin(np.asarray(columns), np.asarray(list(metadata)))
    total = a.sum(axis=1)
    keep = total > 0
    if not keep.any():
        return float("nan")
    return float((a[keep][:, meta].sum(axis=1) / total[keep]).mean())
