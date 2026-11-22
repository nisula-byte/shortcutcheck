"""Figures and the numbers behind them for the paper draft.

Reads only committed report tables (no data access) and writes PDF figures plus a CSV of every
plotted value to the output folder.

Usage: python scripts/paper_figures.py [OUT_DIR]   (default reports/paper)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
ALGO_STYLE = {  # fixed order: colour slot and marker shape (secondary encoding)
    "lr": ("#2a78d6", "o", "LR"),
    "svm": ("#eb6834", "s", "SVM"),
    "rf": ("#1baf7a", "^", "RF"),
    "lgbm": ("#eda100", "D", "LightGBM"),
    "mlp": ("#e87ba4", "v", "MLP"),
}
SPLIT_LABEL = {
    "rq12-DR-AG": "DR-AG",
    "rq12-DR-AG_deb": "DR-AG$_{deb}$",
    "rq12-VS-AG": "VS-AG",
    "rq12-VS-AG_deb": "VS-AG$_{deb}$",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#6b6a64",
        "axes.labelcolor": "#2b2b29",
        "xtick.color": "#4a4945",
        "ytick.color": "#4a4945",
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def _bars(ax, groups, series, values, colors, ymax=None, labelled=None):
    n = len(series)
    width = 0.8 / n
    for i, (name, color) in enumerate(zip(series, colors, strict=True)):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [values.get((grp, name), float("nan")) for grp in groups]
        ax.bar(xs, ys, width * 0.92, color=color, label=name)
        if labelled is not None and name not in labelled:
            continue
        for x, y in zip(xs, ys, strict=True):
            if y == y:  # not NaN
                ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(groups)))
    if ymax:
        ax.set_ylim(0, ymax)


def fig_debiasing(out: Path) -> pd.DataFrame:
    """MR4 on debiased-test inputs by training arm: our protocol (RQ2) and Dada's (Addendum B)."""
    records = []
    for folder, protocol, runs in (
        ("rq2", "re-split", {"DR-AG": "DR", "VS-AG": "VS"}),
        ("rq2-dada", "Dada", {"VS-AG": "VS"}),
    ):
        rows = pd.read_csv(REPORTS / folder / "rows.csv")
        r = rows[(rows["role"] == "h2-deb") & (rows["mr"] == "MR4") & (rows["stratum"] == "all")]
        for family, short in runs.items():
            for arm, suffix in (
                ("published", ""),
                ("size-matched", "-size-matched"),
                ("debiased", "_deb"),
            ):
                part = r[r["run"] == f"{family}{suffix}"]
                records.append(
                    {
                        "group": f"{short} ({protocol})",
                        "arm": arm,
                        "mr4": part["rate"].mean(),
                        "models": len(part),
                    }
                )
    data = pd.DataFrame(records)
    groups = list(dict.fromkeys(data["group"]))
    arms = ["published", "size-matched", "debiased"]
    vals = {(g, a): v for g, a, v in data[["group", "arm", "mr4"]].itertuples(index=False)}
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    _bars(ax, groups, arms, vals, [BLUE, GRAY, ORANGE], ymax=0.27)
    ax.set_xticklabels(groups)
    ax.set_ylabel("MR4 violation rate")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_debiasing.pdf")
    plt.close(fig)
    return data


def fig_fix(out: Path) -> pd.DataFrame:
    """H4: MR4 on test inputs, baseline vs fixed vs duplicate control."""
    rows = pd.read_csv(REPORTS / "addendum_d" / "h4_rows.csv")
    data = rows.groupby(["split", "arm"], sort=False)["rate_MR4"].mean().reset_index()
    data["group"] = data["split"].map(SPLIT_LABEL)
    groups = [SPLIT_LABEL[s] for s in SPLIT_LABEL]
    arms = {"baseline": "baseline", "duplicate": "duplicate control", "augmented": "fixed"}
    vals = {
        (SPLIT_LABEL[s], arms[a]): v
        for s, a, v in data[["split", "arm", "rate_MR4"]].itertuples(index=False)
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    _bars(
        ax,
        groups,
        list(arms.values()),
        vals,
        [BLUE, GRAY, ORANGE],
        ymax=0.25,
        labelled={"baseline", "fixed"},
    )
    ax.set_xticklabels(groups)
    ax.set_ylabel("MR4 violation rate (test)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_fix.pdf")
    plt.close(fig)
    return data


def fig_monitor(out: Path) -> pd.DataFrame:
    """H5b: MR4 at monitoring time and PSI against 2015-16 F1 decay (un-injected models)."""
    pool = pd.read_csv(REPORTS / "addendum_d" / "h5b_pool.csv")
    part = pool[pool["skew"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, col, label in (
        (axes[0], "mr4_monitor", "MR4 violation rate on 2015–16 inputs"),
        (axes[1], "psi", "PSI of predicted scores, 2015–16 vs validation"),
    ):
        for alg, (color, marker, name) in ALGO_STYLE.items():
            g = part[part["algorithm"] == alg]
            ax.scatter(
                g[col],
                g["decay"],
                s=14,
                c=color,
                marker=marker,
                label=name,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_xlabel(label)
    axes[0].set_ylabel("F1 decay to 2015–16")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_monitor.pdf")
    plt.close(fig)
    return part[["model", "algorithm", "mr4_monitor", "psi", "decay"]]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "paper"
    out.mkdir(parents=True, exist_ok=True)
    fig_debiasing(out).to_csv(out / "fig_debiasing.csv", index=False)
    fig_fix(out).to_csv(out / "fig_fix.csv", index=False)
    fig_monitor(out).to_csv(out / "fig_monitor.csv", index=False)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
