"""Apply an MR to a fitted model and record what happens to each input (roadmap step 2.1).

The runner never needs labels. For each input it records the malware probability before and
after the perturbation, whether the MR actually changed the input, and whether the new values
are valid. Violation statistics are computed only over rows that were changed, valid *and* realistic
(step 2.4: close enough to a real training app). The unfiltered rate is kept for the
Revision v2 ablation. Confidence intervals come with the metrics step (2.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from shortcutcheck.mrs.base import MetamorphicRelation, Violation, changed_columns, changed_rows
from shortcutcheck.mrs.realism import RealismCheck


class ProbabilisticModel(Protocol):
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


class ContractViolation(AssertionError):
    """An MR changed a column it did not declare. The MR is wrong, not the model."""


def model_inputs(model: ProbabilisticModel, x: pd.DataFrame) -> pd.DataFrame:
    """The columns the model was trained on, in its order (scikit-learn ``feature_names_in_``).

    This is why a behaviour-only model cannot react to a metadata MR: it never sees the
    columns the MR changes.
    """
    names = getattr(model, "feature_names_in_", None)
    return x if names is None else x[list(names)]


def malware_probability(model: ProbabilisticModel, x: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(model_inputs(model, x)), dtype=float)
    return proba[:, 1] if proba.ndim == 2 else proba


@dataclass(frozen=True)
class MRRun:
    """Per-input outcome of one MR on one model."""

    mr_id: str
    p_before: np.ndarray
    p_after: np.ndarray
    perturbed: np.ndarray
    valid: np.ndarray
    threshold: float = 0.5
    seed: int = 0
    direction: Violation = "any"
    realistic: np.ndarray | None = None  # None = realism check not applied
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def _realistic(self) -> np.ndarray:
        return np.ones(len(self.p_before), bool) if self.realistic is None else self.realistic

    @property
    def counted_unfiltered(self) -> np.ndarray:
        """Changed by the MR and valid, before the realism filter (ablation)."""
        return np.asarray(self.perturbed & self.valid, dtype=bool)

    @property
    def counted(self) -> np.ndarray:
        """Rows that enter the violation statistics: changed, valid and realistic."""
        return np.asarray(self.counted_unfiltered & self._realistic, dtype=bool)

    @property
    def n_unrealistic(self) -> int:
        """Changed and valid rows excluded by the realism check."""
        return int((self.counted_unfiltered & ~self._realistic).sum())

    @property
    def flipped(self) -> np.ndarray:
        before = self.p_before >= self.threshold
        after = self.p_after >= self.threshold
        return np.asarray(before != after, dtype=bool)

    @property
    def violated(self) -> np.ndarray:
        """Flips that break the relation: any flip, or only malware -> goodware (MR6)."""
        if self.direction == "any":
            return self.flipped
        before = self.p_before >= self.threshold
        after = self.p_after >= self.threshold
        return np.asarray(before & ~after, dtype=bool)

    @property
    def n_inputs(self) -> int:
        return len(self.p_before)

    @property
    def n_counted(self) -> int:
        return int(self.counted.sum())

    @property
    def n_violations(self) -> int:
        return int((self.violated & self.counted).sum())

    @property
    def violation_rate(self) -> float:
        """Share of counted inputs that violate the relation (NaN if none counted)."""
        return self.n_violations / self.n_counted if self.n_counted else float("nan")

    @property
    def violation_rate_unfiltered(self) -> float:
        """Violation rate without the realism filter (Revision v2 ablation)."""
        n = int(self.counted_unfiltered.sum())
        return int((self.violated & self.counted_unfiltered).sum()) / n if n else float("nan")

    @property
    def mean_abs_delta(self) -> float:
        """Mean |change in malware probability| over counted inputs (NaN if none counted)."""
        if not self.n_counted:
            return float("nan")
        return float(np.abs(self.p_after - self.p_before)[self.counted].mean())


def run_mr(
    model: ProbabilisticModel,
    mr: MetamorphicRelation,
    x: pd.DataFrame,
    *,
    seed: int,
    threshold: float = 0.5,
    realism: RealismCheck | None = None,
) -> MRRun:
    """Perturb ``x`` with a fitted ``mr`` and compare the model's outputs. Needs no labels.

    With a fitted ``realism`` check, perturbed rows far from every real training app are
    excluded from the statistics (and counted in ``n_unrealistic``).
    """
    rng = np.random.default_rng(seed)
    p_before = malware_probability(model, x)
    x_new = mr.transform(x, rng=rng, predicted=(p_before >= threshold).astype(int))
    extra = changed_columns(x, x_new) - set(mr.columns_touched)
    if extra:
        raise ContractViolation(f"{mr.mr_id} changed undeclared columns: {sorted(extra)}")
    perturbed = changed_rows(x, x_new, mr.columns_touched)
    realistic = None
    if realism is not None:
        realistic = np.ones(len(x), dtype=bool)
        if perturbed.any():
            realistic[perturbed] = realism.is_realistic(x_new.loc[perturbed])
    return MRRun(
        mr_id=mr.mr_id,
        p_before=p_before,
        p_after=malware_probability(model, x_new),
        perturbed=perturbed,
        valid=mr.is_valid(x_new),
        realistic=realistic,
        threshold=threshold,
        seed=seed,
        direction=mr.violation,
    )
