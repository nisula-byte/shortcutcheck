"""Addendum B (PREREGISTRATION-ADDENDUM-B.md): the H2 reversal under Dada's own protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shortcutcheck.analysis.stats import cliffs_delta, holm, wilcoxon_greater

FILE = "PREREGISTRATION-ADDENDUM-B.md"
CONFIG = "configs/experiments/rq2-dada.yaml"
FAMILIES = ("DR-AG", "VS-AG")
CORE_MRS = ("MR1", "MR2", "MR3", "MR4")
ALPHA = 0.05
MIN_COUNTED = 30
COMPARISONS = {"published": "{f}", "size-matched": "{f}-size-matched"}
INPUTS = ("h2-deb", "h2-biased")
DIRECTIONS = ("reversal", "original")  # d = biased - debiased; reversal: d < 0


def tests(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[(rows["stratum"] == "all") & (rows["feature_set"] == "full")]
    out = []
    for comparison, pattern in COMPARISONS.items():
        for inputs in INPUTS:
            for family in FAMILIES:
                b_rows = r[(r["run"] == pattern.format(f=family)) & (r["role"] == inputs)]
                d_rows = r[(r["run"] == f"{family}_deb") & (r["role"] == inputs)]
                for mr in CORE_MRS:
                    b = b_rows[b_rows["mr"] == mr][["algorithm", "seed", "rate"]]
                    d = d_rows[d_rows["mr"] == mr][["algorithm", "seed", "rate"]]
                    if b.empty or d.empty:
                        continue
                    p = b.merge(d, on=["algorithm", "seed"], suffixes=("_b", "_d")).dropna()
                    diff = (p["rate_b"] - p["rate_d"]).to_numpy()
                    for direction in DIRECTIONS:
                        signed = -diff if direction == "reversal" else diff
                        w = wilcoxon_greater(signed.tolist())
                        out.append(
                            {
                                "comparison": comparison,
                                "inputs": inputs,
                                "family": family,
                                "mr": mr,
                                "direction": direction,
                                "n_pairs": w.n_pairs,
                                "mean_biased": p["rate_b"].mean(),
                                "mean_debiased": p["rate_d"].mean(),
                                "p": w.p_greater,
                                "p_two_sided": w.p_two_sided,
                                "cliffs_delta": cliffs_delta(p["rate_b"], p["rate_d"]),
                            }
                        )
    t = pd.DataFrame(out)
    if t.empty:
        return t
    t["p_holm"] = np.nan
    for _, g in t.groupby(["comparison", "inputs", "direction"], sort=False):
        t.loc[g.index, "p_holm"] = holm(g["p"].tolist())
    t["significant"] = t["p_holm"] < ALPHA
    return t


def testable(rows: pd.DataFrame, family: str) -> tuple[bool, int]:
    """Every published and debiased model of the family has >= MIN_COUNTED counted MR4 inputs."""
    r = rows[
        (rows["stratum"] == "all")
        & (rows["mr"] == "MR4")
        & (rows["role"] == "h2-deb")
        & rows["run"].isin([family, f"{family}_deb"])
    ]
    if len(r) < 50:
        return False, int(r["n_counted"].min()) if len(r) else 0
    least = int(r["n_counted"].min())
    return least >= MIN_COUNTED, least


@dataclass(frozen=True)
class Family:
    family: str
    testable: bool
    least_counted: int
    outcome: str  # reversal replicates | original direction | no difference | not testable
    robust: bool


@dataclass(frozen=True)
class BResult:
    families: list[Family]
    table: pd.DataFrame

    @property
    def verdict(self) -> str:
        by = {f.family: f for f in self.families}
        vs, dr = by["VS-AG"], by["DR-AG"]
        if not vs.testable:
            return "NOT TESTABLE"
        replicated = [f for f in self.families if f.testable and f.outcome == "reversal replicates"]
        testable_n = sum(f.testable for f in self.families)
        if vs.outcome == "reversal replicates" and (not dr.testable or dr in replicated):
            return "REPLICATES"
        if len(replicated) == 1 and testable_n == 2:
            return "PARTIALLY REPLICATES"
        return "DOES NOT REPLICATE"


def _sig(t: pd.DataFrame, comparison: str, family: str, direction: str) -> bool:
    x = t[
        (t["comparison"] == comparison)
        & (t["inputs"] == "h2-deb")
        & (t["family"] == family)
        & (t["mr"] == "MR4")
        & (t["direction"] == direction)
    ]
    return bool(len(x) == 1 and x["significant"].iloc[0])


def judge(rows: pd.DataFrame) -> BResult:
    t = tests(rows)
    fams = []
    for family in FAMILIES:
        ok, least = testable(rows, family)
        if not ok:
            outcome = "not testable"
        elif _sig(t, "published", family, "reversal"):
            outcome = "reversal replicates"
        elif _sig(t, "published", family, "original"):
            outcome = "original direction"
        else:
            outcome = "no difference"
        robust = ok and _sig(t, "size-matched", family, "reversal")
        fams.append(Family(family, ok, least, outcome, robust))
    return BResult(fams, t)


def _f(v: Any, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def _p(v: Any) -> str:
    return "–" if pd.isna(v) else (f"{v:.1e}" if v < 0.001 else f"{v:.3f}")


def render(result: BResult, h2_original: pd.DataFrame, head: str) -> str:
    lines = [
        "# Addendum B: the H2 reversal under Dada's own train/test protocol",
        "",
        head,
        "",
        f"**Overall: {result.verdict}.**",
        "",
        "| Family | Testable (min counted MR4 inputs) | Outcome (MR4, published vs debiased, "
        "Dada's debiased test file) | Robust to training size |",
        "| --- | --- | --- | --- |",
    ]
    for f in result.families:
        lines.append(
            f"| {f.family} | {'yes' if f.testable else 'no'} ({f.least_counted}) | "
            f"{f.outcome} | {'yes' if f.robust else 'no'} |"
        )
    lines += [
        "",
        "## Every test",
        "",
        "d = rate(biased) − rate(debiased) over pairs matched by algorithm and seed; "
        "one-sided Wilcoxon per direction; Holm per (comparison, inputs, direction).",
        "",
        "| Comparison | Inputs | Family | MR | Direction | Pairs | Mean biased | "
        "Mean debiased | p | p (Holm) | p (two-sided) | Cliff's δ |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in result.table.itertuples():
        lines.append(
            f"| {t.comparison} | {t.inputs} | {t.family} | {t.mr} | {t.direction} | "
            f"{t.n_pairs} | {_f(t.mean_biased)} | {_f(t.mean_debiased)} | {_p(t.p)} | "
            f"{_p(t.p_holm)} | {_p(t.p_two_sided)} | {_f(t.cliffs_delta, 2)} |"
        )
    if len(h2_original):
        lines += [
            "",
            "## The original H2 code applied to these results (for comparability)",
            "",
            "| Comparison | Inputs | Family | MR | p (one-sided, biased > debiased, Holm) |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for t in h2_original.itertuples():
            lines.append(f"| {t.comparison} | {t.inputs} | {t.family} | {t.mr} | {_p(t.p_holm)} |")
    return "\n".join(lines) + "\n"
