"""Exploratory: within-algorithm Spearman rho between the MR4 monitor and 2015-16 decay.

Not pre-registered. Uses only the per-model tables written by addenda D and E (no test
partition is read). For each dataset and algorithm: Spearman rho over that algorithm's models
(30 per algorithm: 6 settings x 5 seeds; AndroCT un-injected subset only).

Usage: python scripts/per_algorithm_monitor_exploratory.py
Writes reports/exploratory/per_algorithm_monitor.md and .csv.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from shortcutcheck.analysis.stats import spearman
from shortcutcheck.config import load_config
from shortcutcheck.training.baselines import _setup_mlflow

OUT = Path("reports/exploratory")


def tables() -> dict[str, pd.DataFrame]:
    androct = pd.read_csv("reports/addendum_d/h5b_pool.csv")
    androct = androct[androct["skew"] == 0]
    azl = pd.read_csv("reports/addendum_e/e_pool.csv")
    return {"AndroCT": androct, "AZL": azl}


def main() -> None:
    rows = []
    for name, t in tables().items():
        t = t.dropna(subset=["mr4_monitor", "decay"])
        for algorithm, g in t.groupby("algorithm", sort=True):
            rows.append(
                {
                    "dataset": name,
                    "algorithm": algorithm,
                    "n_models": len(g),
                    "rho": spearman(g["mr4_monitor"], g["decay"]),
                }
            )
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_algorithm_monitor.csv", index=False)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    _setup_mlflow(load_config(), "exploratory-per-algorithm-monitor")
    with mlflow.start_run(run_name="per-algorithm-mr4") as run:
        mlflow.set_tags({"git_commit": commit, "status": "exploratory"})
        for r in rows:
            mlflow.log_metric(f"{r['dataset']}_{r['algorithm']}_rho", float(r["rho"]))
        run_id = run.info.run_id
    lines = [
        "# Exploratory: within-algorithm rho, MR4 monitor vs 2015-16 decay (not pre-registered)",
        "",
        "| Dataset | Algorithm | n | rho |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['dataset']} | {r['algorithm']} | {r['n_models']} | {r['rho']:.3f} |" for r in rows
    ]
    lines += ["", f"Script commit {commit}; MLflow run {run_id}."]
    (OUT / "per_algorithm_monitor.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
