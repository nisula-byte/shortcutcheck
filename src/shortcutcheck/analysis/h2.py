"""H2 (RQ2, roadmap step 3.4), judged exactly as PREREGISTRATION.md states it.

Input: ``reports/rq2/rows.csv`` from ``make rq2`` (full-feature models, ``all`` stratum).

For each family (DR-AG, VS-AG), input set (``h2-deb`` primary, ``h2-biased`` secondary) and
core MR, the 25 paired differences d = rate(biased model) - rate(debiased model), matched by
algorithm and seed and scored on identical inputs, are tested with a one-sided Wilcoxon
signed-rank test (d > 0). Holm correction runs over the core-MR tests of both families within
each (comparison, input set) family of tests. Cliff's delta is reported next to every p-value.

* Primary comparison: published biased models vs debiased models, on ``h2-deb`` inputs.
* A family **supports H2** if MR4 has a Holm-adjusted p < ``ALPHA``.
* **Verdict**: supported if both families support it, partially supported if one, else not.
* **Robust to training size**: per family, the same MR4 test with the size-matched biased
  models in place of the published ones is also significant after Holm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from shortcutcheck.analysis.stats import cliffs_delta, holm, wilcoxon_greater

FAMILIES = ("DR-AG", "VS-AG")
CORE_MRS = ("MR1", "MR2", "MR3", "MR4")
PRIMARY_MR = "MR4"
ALPHA = 0.05
COMPARISONS = {"published": "{f}", "size-matched": "{f}-size-matched"}
INPUTS = ("h2-deb", "h2-biased")
EXPECTED_PAIRS = 25


def h2_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """Every paired test, with Holm-adjusted p-values per (comparison, inputs) family."""
    r = rows[(rows["stratum"] == "all") & (rows["feature_set"] == "full")]
    out = []
    for comparison, pattern in COMPARISONS.items():
        for inputs in INPUTS:
            for family in FAMILIES:
                biased = r[(r["run"] == pattern.format(f=family)) & (r["role"] == inputs)]
                debiased = r[(r["run"] == f"{family}_deb") & (r["role"] == inputs)]
                for mr in CORE_MRS:
                    b = biased[biased["mr"] == mr][["algorithm", "seed", "rate"]]
                    d = debiased[debiased["mr"] == mr][["algorithm", "seed", "rate"]]
                    if b.empty or d.empty:
                        continue
                    paired = b.merge(d, on=["algorithm", "seed"], suffixes=("_b", "_d")).dropna()
                    w = wilcoxon_greater((paired["rate_b"] - paired["rate_d"]).tolist())
                    out.append(
                        {
                            "comparison": comparison,
                            "inputs": inputs,
                            "family": family,
                            "mr": mr,
                            "n_pairs": w.n_pairs,
                            "n_nonzero": w.n_nonzero,
                            "mean_biased": paired["rate_b"].mean(),
                            "mean_debiased": paired["rate_d"].mean(),
                            "median_difference": (paired["rate_b"] - paired["rate_d"]).median(),
                            "statistic": w.statistic,
                            "p_greater": w.p_greater,
                            "p_two_sided": w.p_two_sided,
                            "cliffs_delta": cliffs_delta(paired["rate_b"], paired["rate_d"]),
                        }
                    )
    tests = pd.DataFrame(out)
    if tests.empty:
        return tests
    tests["p_holm"] = float("nan")
    for _, g in tests.groupby(["comparison", "inputs"], sort=False):
        tests.loc[g.index, "p_holm"] = holm(g["p_greater"].tolist())
    tests["significant"] = tests["p_holm"] < ALPHA
    return tests


@dataclass(frozen=True)
class H2Result:
    tests: pd.DataFrame
    supports: dict[str, bool]  # family -> primary MR4 test significant
    robust: dict[str, bool]  # family -> size-matched MR4 test significant

    @property
    def verdict(self) -> str:
        held = sum(self.supports.values())
        if held == len(FAMILIES):
            return "SUPPORTED"
        return "PARTIALLY SUPPORTED" if held else "NOT SUPPORTED"


def _sig(tests: pd.DataFrame, comparison: str, family: str) -> bool:
    t = tests[
        (tests["comparison"] == comparison)
        & (tests["inputs"] == "h2-deb")
        & (tests["family"] == family)
        & (tests["mr"] == PRIMARY_MR)
    ]
    return bool(len(t) == 1 and t["significant"].iloc[0])


def judge_h2(rows: pd.DataFrame) -> H2Result:
    tests = h2_tests(rows)
    if tests.empty:
        return H2Result(tests, dict.fromkeys(FAMILIES, False), dict.fromkeys(FAMILIES, False))
    return H2Result(
        tests,
        {f: _sig(tests, "published", f) for f in FAMILIES},
        {f: _sig(tests, "size-matched", f) for f in FAMILIES},
    )


def _f(v: Any, d: int = 3) -> str:
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def _p(v: Any) -> str:
    return "–" if pd.isna(v) else (f"{v:.2e}" if v < 0.001 else f"{v:.3f}")


def render_h2(result: H2Result, rows: pd.DataFrame, header: str) -> str:
    lines = [
        "# RQ2: does debiased training lower violation rates? (H2, roadmap step 3.4)",
        "",
        header,
        "",
        f"**H2: {result.verdict}.**",
        "",
        "| Family | MR4 significant (published vs debiased, debiased-test inputs) "
        "| Robust to training size (size-matched vs debiased) |",
        "| --- | --- | --- |",
    ]
    for f in FAMILIES:
        lines.append(
            f"| {f} | {'yes' if result.supports[f] else 'no'} | "
            f"{'yes' if result.robust[f] else 'no'} |"
        )
    lines += [
        "",
        "## Every test",
        "",
        "One-sided Wilcoxon signed-rank (biased > debiased) over pairs matched by algorithm and "
        "seed; Holm across core MRs and both families within each comparison and input set. "
        "Cliff's delta: +1 = every biased rate above every debiased rate.",
        "",
        "| Comparison | Inputs | Family | MR | Pairs (non-zero) | Mean biased | Mean debiased "
        "| p (one-sided) | p (Holm) | p (two-sided) | Cliff's delta |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in result.tests.itertuples():
        lines.append(
            f"| {t.comparison} | {t.inputs} | {t.family} | {t.mr} | {t.n_pairs} ({t.n_nonzero}) "
            f"| {_f(t.mean_biased)} | {_f(t.mean_debiased)} | {_p(t.p_greater)} "
            f"| {_p(t.p_holm)} | {_p(t.p_two_sided)} | {_f(t.cliffs_delta, 2)} |"
        )
    short = (
        result.tests[result.tests["n_pairs"] < EXPECTED_PAIRS]
        if len(result.tests)
        else pd.DataFrame()
    )
    if len(short):
        lines += [
            "",
            f"Tests with fewer than {EXPECTED_PAIRS} pairs (a model had no counted "
            "input on that MR): "
            + ", ".join(
                f"{t.comparison}/{t.inputs}/{t.family}/{t.mr} ({t.n_pairs})"
                for t in short.itertuples()
            )
            + ".",
        ]
    cross = rows[(rows["role"] == "cross-family") & (rows["stratum"] == "all")]
    if len(cross):
        lines += [
            "",
            "## Cross-family (descriptive, not tested)",
            "",
            "| Trained on | Scored on | Algorithm | MR | Mean rate |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for (tr, ev, a, mr), g in cross.groupby(
            ["run", "eval_split", "algorithm", "mr"], sort=False
        ):
            lines.append(f"| {tr} | {ev} | {a} | {mr} | {_f(g['rate'].mean())} |")
    return "\n".join(lines) + "\n"
