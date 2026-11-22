"""Descriptive: means and standard deviations behind the paper's Tables 9-12.

Reads the per-model rows written by addenda D and E (no test partition is read) and writes
reports/paper/variance.md. SD is the sample standard deviation (ddof=1).

- Table 9 (H4): per dataset and arm, over the 25 algorithm-seed models.
- Table 10 (H4 per algorithm): per dataset, algorithm and arm, over the 5 seeds.
- Table 11 (H5a): 2015-16 decay per algorithm and arm, over the 5 seeds; overall over 25.
- Table 12 (Addendum E, AZL): per arm, over the 25 models.
- Counts of training runs per experiment (abstract and Section IV-B).

Usage: python scripts/paper_variance.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("reports/paper")


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def block(frame: pd.DataFrame, keys: list[str], cols: list[str]) -> list[str]:
    agg = frame.groupby(keys)[cols].agg(["mean", "std"])
    header = "| " + " | ".join(keys + cols) + " |"
    lines = [header, "|" + "---|" * (len(keys) + len(cols))]
    for idx, row in agg.iterrows():
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(i) for i in idx_t] + [fmt(row[(c, "mean")], row[(c, "std")]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def runs() -> list[str]:
    def n(path: str, cols: list[str]) -> int:
        return len(pd.read_csv(path)[cols].drop_duplicates())

    key = ["train_split", "arm", "algorithm", "feature_set", "seed"]
    counts = {
        "RQ1 (biased datasets, full and behavior-only)": n("reports/rq1/detection.csv", key),
        "RQ2 (published, size-matched, debiased)": n("reports/rq2/detection.csv", ["run", *key]),
        "RQ3 pool (AndroCT)": len(pd.read_csv("reports/rq3/pool_models.csv")),
        "Addendum B (Dada protocol)": n("reports/rq2-dada/detection.csv", ["run", *key]),
        "Addendum C (proof of concept)": len(pd.read_csv("reports/addendum_c/rows.csv")),
        "Addendum D, H4": len(pd.read_csv("reports/addendum_d/h4_rows.csv")),
        "Addendum D, H5a": len(pd.read_csv("reports/addendum_d/h5a_rows.csv")),
        "Addendum E (AZL pool, arms; AndroCT behavior-only)": len(
            pd.read_csv("reports/addendum_e/e_pool.csv")
        )
        + len(pd.read_csv("reports/addendum_e/e_arms.csv"))
        + len(pd.read_csv("reports/addendum_e/e_androct_behaviour.csv")),
        "Addendum F (mechanism)": len(pd.read_csv("reports/addendum_f/f_rows.csv")),
    }
    lines = ["| Experiment | Training runs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines.append(f"| Total | {sum(counts.values())} |")
    return lines


def main() -> None:
    h4 = pd.read_csv("reports/addendum_d/h4_rows.csv")
    h5 = pd.read_csv("reports/addendum_d/h5a_rows.csv")
    e = pd.read_csv("reports/addendum_e/e_arms.csv")
    lines = ["# Means ± SD behind Tables 9-12", "", "## Table 9 (over 25 models)", ""]
    lines += block(h4, ["split", "arm"], ["rate_MR4", "f1", "contrary_acc"])
    lines += ["", "## Table 10 (over 5 seeds)", ""]
    lines += block(h4, ["split", "algorithm", "arm"], ["rate_MR4", "f1"])
    lines += ["", "## Table 11 (over 5 seeds; overall over 25)", ""]
    lines += block(h5, ["algorithm", "arm"], ["decay_1516"])
    lines += [""] + block(h5, ["arm"], ["decay_1516"])
    lines += ["", "## Table 12 (over 25 models)", ""]
    lines += block(e, ["arm"], ["f1_val", "f1_holdout", "decay_1516", "decay_1719"])
    lines += ["", "## Training runs per experiment", ""] + runs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
