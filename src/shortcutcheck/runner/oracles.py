"""Oracle models with known answers (roadmap step 2.5).

If the runner is right, these models produce violation rates we can work out in advance:

* ``Stump(column)``: a one-split decision tree on a single column (``year`` or ``apk_size``),
  fitted on training labels. It is a pure shortcut. Its expected flip rate under the MR that
  changes that column can be computed exactly from the MR's sampling rule (``expected_*``).
  Under MRs that do not touch the column it must flip exactly 0 times.
* ``BehaviourOnly``: a random forest that never sees metadata. It must flip exactly 0 times
  under MR1, MR3 and MR4.
* ``Constant``: always the training malware share. It must flip exactly 0 times under every MR.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from shortcutcheck.mrs.core import MetadataSwap, SizeScale, YearShift


class Stump:
    def __init__(self, column: str) -> None:
        self.column = column
        self.feature_names_in_ = np.array([column])
        self._tree = DecisionTreeClassifier(max_depth=1, random_state=0)
        self.threshold = float("nan")
        self.left_class = 0
        self.right_class = 0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Self:
        values = pd.to_numeric(x[self.column], errors="coerce").fillna(0.0).to_frame()
        self._tree.fit(values, np.asarray(y, dtype=int))
        tree = self._tree.tree_
        if tree.node_count < 3:
            raise ValueError(f"no useful split on {self.column}")
        self.threshold = float(tree.threshold[0])
        p_left = tree.value[tree.children_left[0]][0]
        p_right = tree.value[tree.children_right[0]][0]
        self.left_class = int(np.argmax(p_left))
        self.right_class = int(np.argmax(p_right))
        return self

    def predict_class(self, values: np.ndarray) -> np.ndarray:
        """Class from the raw column value (scikit-learn sends value <= threshold left)."""
        return np.where(values <= self.threshold, self.left_class, self.right_class)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        v = pd.to_numeric(x[self.column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        p = self.predict_class(v).astype(float)
        return np.column_stack([1 - p, p])


class Constant:
    feature_names_in_ = np.array([], dtype=object)

    def __init__(self) -> None:
        self.p = 0.5

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Self:
        self.p = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.p, self.p], (len(x), 1))


# ---- exact expected flip rates -------------------------------------------------------


def expected_mr1(stump: Stump, mr: YearShift, years: np.ndarray) -> np.ndarray:
    """Per input: probability that MR1 flips a year stump.

    MR1 draws uniformly from the valid training years, excluding the app's own year.
    """
    pool = mr.valid_years
    pool_class = stump.predict_class(pool)
    own_class = stump.predict_class(years)
    out = np.empty(len(years))
    for i, (y, c) in enumerate(zip(years, own_class, strict=True)):
        candidates = pool != y
        out[i] = float(np.mean(pool_class[candidates] != c))
    return out


def expected_mr3(stump: Stump, mr: SizeScale, sizes: np.ndarray) -> np.ndarray:
    """Per input: probability that MR3 moves a size across the stump's threshold.

    MR3 multiplies by f with ln f uniform on [ln a, ln b], then clips to the training range.
    Clipping cannot cross a threshold that lies inside the range, so only the factor matters.
    """
    a, b = mr.factor
    span = np.log(b) - np.log(a)
    t = stump.threshold
    s = np.asarray(sizes, dtype=float)
    ratio = np.log(t / s)
    up = np.clip((np.log(b) - ratio) / span, 0.0, 1.0)  # P(s*f > t), for s <= t
    down = np.clip((ratio - np.log(a)) / span, 0.0, 1.0)  # P(s*f <= t), for s > t
    crosses = np.where(s <= t, up, down)
    differ = stump.left_class != stump.right_class
    return crosses if differ else np.zeros_like(crosses)


def expected_mr4(stump: Stump, mr: MetadataSwap, values: np.ndarray) -> np.ndarray:
    """Per input: probability that MR4 flips a stump on one metadata column.

    MR4 copies the metadata of a uniformly drawn training donor from the class opposite to the
    model's own prediction.
    """
    j = list(mr.columns_touched).index(stump.column)
    own = stump.predict_class(values)
    rate = {c: float(np.mean(stump.predict_class(mr.donors[1 - c][:, j]) != c)) for c in (0, 1)}
    return np.array([rate[int(c)] for c in own])
