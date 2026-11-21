"""Figures for RQ1-RQ3, written as SVG with fixed metadata so reruns are byte-identical."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "shortcutcheck"
matplotlib.rcParams["font.size"] = 9
ALGORITHM_ORDER = ["lr", "svm", "rf", "lgbm", "mlp"]


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    return path


def rq1_violation_rates(rows: pd.DataFrame, path: Path, mrs: tuple[str, ...]) -> Path:
    """Mean violation rate (min-max over seeds as error bars) per split, algorithm and MR."""
    r = rows[(rows["stratum"] == "all") & rows["mr"].isin(mrs)]
    splits = list(dict.fromkeys(r["train_split"]))
    present = [m for m in mrs if m in set(r["mr"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.2), squeeze=False)
    width = 0.8 / max(1, 2 * len(present))
    for ax, split in zip(axes[0], splits, strict=True):
        s = r[r["train_split"] == split]
        for i, (mr, fs) in enumerate((m, f) for m in present for f in ("full", "behaviour")):
            g = s[(s["mr"] == mr) & (s["feature_set"] == fs)].groupby("algorithm")["rate"]
            stats = g.agg(["mean", "min", "max"]).reindex(ALGORITHM_ORDER)
            x = [j + (i - len(present)) * width + width / 2 for j in range(len(ALGORITHM_ORDER))]
            ax.bar(
                x,
                stats["mean"],
                width,
                yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
                label=f"{mr} {fs}",
                hatch="" if fs == "full" else "//",
                capsize=1.5,
            )
        ax.axhline(0.05, color="grey", linewidth=0.7, linestyle="--")
        ax.set_xticks(range(len(ALGORITHM_ORDER)), ALGORITHM_ORDER)
        ax.set_title(split)
        ax.set_ylabel("violation rate")
    axes[0][-1].legend(fontsize=7, frameon=False)
    return _save(fig, path)


def rq2_pairs(tests_rows: pd.DataFrame, path: Path, mr: str = "MR4") -> Path:
    """Per family: each algorithm-seed pair's MR rate, biased vs debiased (h2-deb inputs)."""
    r = tests_rows[(tests_rows["stratum"] == "all") & (tests_rows["mr"] == mr)]
    r = r[r["role"] == "h2-deb"]
    families = ["DR-AG", "VS-AG"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), squeeze=False)
    for ax, family in zip(axes[0], families, strict=True):
        b = r[r["run"] == family].set_index(["algorithm", "seed"])["rate"]
        d = r[r["run"] == f"{family}_deb"].set_index(["algorithm", "seed"])["rate"]
        joined = pd.concat([b.rename("biased"), d.rename("debiased")], axis=1).dropna()
        for _, row in joined.iterrows():
            ax.plot([0, 1], [row["biased"], row["debiased"]], color="grey", linewidth=0.5)
        ax.scatter([0] * len(joined), joined["biased"], s=8)
        ax.scatter([1] * len(joined), joined["debiased"], s=8)
        ax.set_xticks([0, 1], ["biased", "debiased"])
        ax.set_title(f"{family}, {mr}")
        ax.set_ylabel("violation rate")
    return _save(fig, path)


def rq3_scatter(pool: pd.DataFrame, path: Path, signal: str = "mr4_pre") -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for skew, g in pool.groupby("skew"):
        ax.scatter(g[signal], g["f1_drop"], s=8, label=f"skew {skew:g}")
    ax.set_xlabel(f"{signal} (pre-deployment)")
    ax.set_ylabel("F1 drop (2012-14 to 2017-19)")
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, path)
