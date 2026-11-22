"""Build the CI demo (roadmap 4.2): a small synthetic dataset and two detectors.

- ``models/shortcut.joblib``: LightGBM trained on every feature. In this data malware is old and
  small, so the model learns build year and package size: a deliberately shortcut-prone model.
- ``models/robust.joblib``: logistic regression trained on behaviour features only.

The data are synthetic (no real apps). Usage: ``python make_demo.py`` (writes data/ and models/
next to this file).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
BEHAVIOUR = [f"api_{i}" for i in range(8)]


def apps(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "sha256": [f"demo{seed:02d}{i:05d}" for i in range(n)],
            "label": y,
            "year": np.where(y == 1, rng.integers(2010, 2014, n), rng.integers(2015, 2020, n)),
            "apk_size": np.where(y == 1, rng.normal(2e6, 4e5, n), rng.normal(7e6, 1.5e6, n))
            .clip(1e5)
            .round(),
            "minSdkVersion": 0,
            "targetSdkVersion": 0,
        }
    )
    for j, col in enumerate(BEHAVIOUR):
        p = np.where(y == 1, 0.6 + 0.03 * j, 0.4 - 0.03 * j)
        frame[col] = (rng.random(n) < p).astype(int)
    return frame


def main() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "models").mkdir(exist_ok=True)
    train, deploy = apps(1500, 1), apps(600, 2)
    train.to_csv(HERE / "data" / "train.csv", index=False)
    deploy.drop(columns=["label"]).to_csv(HERE / "data" / "apps.csv", index=False)
    x = train.drop(columns=["sha256", "label"])
    shortcut = LGBMClassifier(n_estimators=200, random_state=11, verbose=-1).fit(x, train["label"])
    robust = LogisticRegression(max_iter=2000).fit(x[BEHAVIOUR], train["label"])
    joblib.dump(shortcut, HERE / "models" / "shortcut.joblib")
    joblib.dump(robust, HERE / "models" / "robust.joblib")
    print(f"wrote demo data and models to {HERE}")


if __name__ == "__main__":
    main()
