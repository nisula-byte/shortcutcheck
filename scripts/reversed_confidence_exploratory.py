"""Exploratory: confidence drop with its sign reversed, as a monitor of decay.

Not pre-registered. Uses only the per-model tables already written by addenda D and E
(no test partition is read). The reversed signal is fixed here before it is computed:
the negative of the pre-specified confidence drop (mean max-probability on validation
minus on deployment inputs), with no tuning.
Raised by a reviewer because confidence drop correlated with decay in the opposite direction.

Usage: python scripts/reversed_confidence_exploratory.py
Writes reports/exploratory/reversed_confidence.md and .csv.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.addenda.addendum_e import adjusted_bootstrap, adjusted_rho
from shortcutcheck.analysis.stats import cluster_bootstrap_rhos, percentile_ci, spearman
from shortcutcheck.config import load_config
from shortcutcheck.training.baselines import _setup_mlflow

RESAMPLES = 2000
SEED = 11
SIGNALS = ("mr4_monitor", "psi", "confidence_reversed")
OUT = Path("reports/exploratory")


def reversed_confidence(table: pd.DataFrame) -> pd.Series:
    """Confidence drop with its sign reversed."""
    return -table["confidence_drop"]


def load() -> dict[str, pd.DataFrame]:
    androct = pd.read_csv("reports/addendum_d/h5b_pool.csv")
    androct = androct[androct["skew"] == 0].copy()  # the 150 un-injected models (H5b subset)
    azl = pd.read_csv("reports/addendum_e/e_pool.csv")
    azl["configuration"] = azl["algorithm"].astype(str) + ":" + azl["setting"].astype(str)
    tables = {"AndroCT 2015-16": androct, "AZL 2015-16": azl}
    for name, t in tables.items():
        t = t.dropna(subset=["mr4_monitor", "psi", "confidence_drop", "decay"]).copy()
        t["confidence_reversed"] = reversed_confidence(t)
        t["_cluster"] = t["configuration"]
        tables[name] = t
    return tables


def analyse(name: str, t: pd.DataFrame) -> list[dict[str, object]]:
    t = t.reset_index(drop=True)
    boot = cluster_bootstrap_rhos(t, list(SIGNALS), "decay", "_cluster", RESAMPLES, SEED)
    rows: list[dict[str, object]] = []
    for s in SIGNALS:
        raw_low, raw_high = percentile_ci(np.asarray(boot[s]))
        adj = adjusted_rho(
            t[s].to_numpy(float), t["decay"].to_numpy(float), t["algorithm"].to_numpy()
        )
        adj_low, adj_high = percentile_ci(adjusted_bootstrap(t, s, resamples=RESAMPLES, seed=SEED))
        rows.append(
            {
                "dataset": name,
                "signal": s,
                "n_models": len(t),
                "rho": spearman(t[s], t["decay"]),
                "rho_low": raw_low,
                "rho_high": raw_high,
                "adj_rho": adj,
                "adj_low": adj_low,
                "adj_high": adj_high,
            }
        )
    return rows


def main() -> None:
    rows = [r for name, t in load().items() for r in analyse(name, t)]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "reversed_confidence.csv", index=False)
    lines = [
        "# Exploratory: confidence drop with sign reversed (not pre-registered)",
        "",
        "confidence_reversed = minus the pre-specified confidence drop, fixed before computing. "
        "Spearman rho with F1 decay to 2015-16; 95% cluster-bootstrap CIs "
        f"({RESAMPLES} resamples, seed {SEED}); adj. = after removing algorithm mean ranks.",
        "",
        "| Dataset | Signal | n | rho | 95% CI | adj. rho | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['signal']} | {r['n_models']} | {r['rho']:.3f} | "
            f"[{r['rho_low']:.3f}, {r['rho_high']:.3f}] | {r['adj_rho']:.3f} | "
            f"[{r['adj_low']:.3f}, {r['adj_high']:.3f}] |"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    _setup_mlflow(load_config(), "exploratory-reversed-confidence")
    with mlflow.start_run(run_name="reversed-confidence") as run:
        mlflow.set_tags({"git_commit": commit, "status": "exploratory"})
        mlflow.log_params({"resamples": RESAMPLES, "seed": SEED, "signal": "minus confidence_drop"})
        for r in rows:
            key = f"{str(r['dataset']).split()[0]}_{r['signal']}"
            mlflow.log_metrics(
                {f"{key}_rho": float(r["rho"]), f"{key}_adj_rho": float(r["adj_rho"])}
            )
        run_id = run.info.run_id
    lines += ["", f"Script commit {commit}; MLflow run {run_id}."]
    (OUT / "reversed_confidence.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
