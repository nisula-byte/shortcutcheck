"""Secondary metamorphic relations MR5 and MR6 (roadmap step 2.3).

Both add exactly one permission an app does not already request. Which permissions may be
added comes only from the official protection levels in ``permissions.py``; fitting on
training data then narrows that list (see ``min_apps`` and, for MR5, goodware-common).

* **MR5** (invariance, asymmetric): give an app the model calls malware one extra *normal*
  permission that is more common in training goodware. Benign extras do not cancel malicious
  code, so the app must stay "malware".
* **MR6** (monotonicity): give any app one extra *dangerous* permission. More access to
  sensitive data cannot make an app less suspicious, so a malware prediction must not become
  goodware. Goodware -> malware is allowed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError
from shortcutcheck.mrs.permissions import mr5_columns, mr6_columns


def _add_one(
    x: pd.DataFrame, pool: list[str], rows: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """For each selected row, set one random pool permission it lacks from 0 to 1."""
    if not pool:
        return x
    held = x[pool].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    candidate = (held == 0) & rows[:, None]
    keys = rng.random(candidate.shape)
    keys[~candidate] = -1.0
    choice = keys.argmax(axis=1)
    change = candidate.any(axis=1)
    for j, col in enumerate(pool):
        hit = change & (choice == j)
        if hit.any():
            values = x[col].to_numpy(copy=True)
            values[hit] = 1
            x[col] = values
    return x


class _AddPermission(MetamorphicRelation):
    role = "secondary"
    eligible: tuple[str, ...] = ()
    rule = "held by {min_apps}+ training apps"

    def __init__(self, columns: Sequence[str], min_apps: int = 10) -> None:
        present = tuple(c for c in self.eligible if c in set(columns))
        if not present:
            raise NotApplicableError(f"{self.mr_id}: none of its permissions are in the data")
        super().__init__(present)
        self.min_apps = min_apps
        self.pool: list[str] = []

    def _keep(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> list[str]:
        held = x_train[list(self.columns_touched)].apply(pd.to_numeric, errors="coerce")
        return [c for c in self.columns_touched if int((held[c] == 1).sum()) >= self.min_apps]

    def _fit(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:
        self.pool = self._keep(x_train, y_train)
        if not self.pool:
            raise NotApplicableError(
                f"{self.mr_id}: no eligible permission is "
                + self.rule.format(min_apps=self.min_apps)
            )
        # Only pool columns can change, so only they are declared and range-checked. Checking
        # every eligible column would mark an app invalid for a value the MR never touched.
        self._columns_touched = tuple(self.pool)
        self._ranges = {c: self.ranges[c] for c in self.pool}


class AddNormalPermission(_AddPermission):
    """MR5."""

    mr_id = "MR5"
    name = "Add a benign permission"
    justification = (
        "Requesting one more low-risk ('normal') permission, of the kind benign apps commonly "
        "request, does not remove malicious code: an app judged malicious must stay malicious."
    )
    eligible = tuple(mr5_columns())
    rule = "held by {min_apps}+ training apps and more common in training goodware than malware"

    def _keep(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> list[str]:
        if y_train is None:
            raise NotApplicableError(
                "MR5 needs training labels to find goodware-common permissions"
            )
        common = super()._keep(x_train, y_train)
        labels = np.asarray(y_train, dtype=int)
        held = x_train[common].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        good = held[labels == 0].mean(axis=0)
        bad = held[labels == 1].mean(axis=0)
        return [c for c, g, b in zip(common, good, bad, strict=True) if g > b]

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        if predicted is None or len(predicted) != len(x):
            raise ValueError("MR5 needs the model's predicted class for every row")
        return _add_one(x, self.pool, np.asarray(predicted, dtype=int) == 1, rng)


class AddDangerousPermission(_AddPermission):
    """MR6."""

    mr_id = "MR6"
    name = "Add a dangerous permission"
    justification = (
        "Requesting one more 'dangerous' permission (access to private data or costly actions) "
        "cannot make an app less suspicious: a malware prediction must not become goodware."
    )
    violation = "to_goodware"
    eligible = tuple(mr6_columns())

    def _transform(
        self, x: pd.DataFrame, *, rng: np.random.Generator, predicted: np.ndarray | None
    ) -> pd.DataFrame:
        return _add_one(x, self.pool, np.ones(len(x), dtype=bool), rng)


def secondary_mrs(columns: Sequence[str], min_apps: int = 10) -> list[MetamorphicRelation]:
    """MR5 and MR6 for the permission columns present (fit them before use)."""
    out: list[MetamorphicRelation] = []
    for cls in (AddNormalPermission, AddDangerousPermission):
        try:
            out.append(cls(columns, min_apps))
        except NotApplicableError:
            continue
    return out
