"""Step 1.4: data audit.

Part 1, every Dada set: rows, unique hashes, label coverage, class balance and year range.

Part 2, RQ3 sufficiency. Dada's ``ACT14``/``ACT17`` (and ``AZL14``/``AZL17``) names are *time
barriers* used to cut one app pool into training and test parts, not the year the apps come
from: ``ACT14`` and ``ACT17`` share most of their hashes. So the RQ3 check pools a family's
apps, reads each app's own ``year`` (the year of its ``dex_date``), and counts goodware and
malware inside the early and late year windows of each configured candidate. A candidate
passes only if every one of the four counts reaches ``rq3_min_per_class_per_year``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from shortcutcheck.data.dada import LABEL, DadaSet, match_sets, read_set

RQ3_YEAR = "year"


@dataclass(frozen=True)
class SetSummary:
    name: str
    rows: int
    unique_hashes: int
    labelled: bool
    unlabelled_rows: int
    goodware: int
    malware: int
    year_min: float | None
    year_max: float | None

    @property
    def malware_ratio(self) -> float | None:
        total = self.goodware + self.malware
        return None if total == 0 else self.malware / total


@dataclass(frozen=True)
class RQ3Verdict:
    candidate: str
    sources: list[str]
    early_years: list[int]
    late_years: list[int]
    early_counts: tuple[int, int]
    late_counts: tuple[int, int]
    minimum: int
    invalid_year_share: float

    @property
    def passes(self) -> bool:
        return bool(self.sources) and (min(*self.early_counts, *self.late_counts) >= self.minimum)


def summarise(frame: pd.DataFrame, name: str, id_column: str) -> SetSummary:
    years = (
        pd.to_numeric(frame[RQ3_YEAR], errors="coerce")
        if RQ3_YEAR in frame.columns
        else pd.Series(dtype=float)
    )
    labelled = LABEL in frame.columns
    labels = frame[LABEL] if labelled else pd.Series(dtype=float)
    return SetSummary(
        name=name,
        rows=len(frame),
        unique_hashes=int(frame[id_column].nunique()),
        labelled=labelled,
        unlabelled_rows=int(labels.isna().sum()) if labelled else len(frame),
        goodware=int((labels == 0).sum()) if labelled else 0,
        malware=int((labels == 1).sum()) if labelled else 0,
        year_min=float(years.min()) if years.notna().any() else None,
        year_max=float(years.max()) if years.notna().any() else None,
    )


def candidate_pool(
    sets: list[DadaSet], candidate: dict[str, Any], id_column: str, dedup: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """All apps of one candidate family, one row per hash, with a 0/1 ``label``.

    A candidate names either ``goodware`` and ``malware`` set globs (class from the set) or
    ``labelled`` set globs (class from the ``.goodmal.csv`` files).
    """
    frames, sources = [], []
    if "labelled" in candidate:
        for pattern in candidate["labelled"]:
            for s in match_sets(sets, pattern):
                frames.append(read_set(s, id_column))
                sources.append(s.name)
    else:
        for key, label in (("goodware", 0), ("malware", 1)):
            for s in match_sets(sets, candidate[key]):
                frames.append(read_set(s, id_column).assign(**{LABEL: label}))
                sources.append(s.name)
    if not frames:
        return pd.DataFrame(columns=[id_column, LABEL, RQ3_YEAR]), []
    pool = pd.concat(frames, ignore_index=True)
    if not dedup:
        return pool, sorted(set(sources))
    pool = pool[pool[LABEL].notna()]
    # A hash listed with both labels is unusable; otherwise keep one row per hash.
    conflicted = pool.groupby(id_column)[LABEL].nunique()
    pool = pool[~pool[id_column].isin(set(conflicted[conflicted > 1].index))]
    pool = pool.drop_duplicates(subset=id_column).reset_index(drop=True)
    return pool, sorted(set(sources))


def valid_years(pool: pd.DataFrame, first_year: int, last_year: int) -> pd.Series:
    years = pd.to_numeric(pool[RQ3_YEAR], errors="coerce")
    return years.where(years.between(first_year, last_year))


def window(pool: pd.DataFrame, years: list[int], first_year: int, last_year: int) -> pd.DataFrame:
    return pool[valid_years(pool, first_year, last_year).isin(years)]


def _counts(frame: pd.DataFrame) -> tuple[int, int]:
    return int((frame[LABEL] == 0).sum()), int((frame[LABEL] == 1).sum())


def rq3_verdicts(
    sets: list[DadaSet],
    candidates: dict[str, dict[str, Any]],
    minimum: int,
    id_column: str,
    first_year: int,
    last_year: int,
) -> tuple[list[RQ3Verdict], dict[str, pd.DataFrame]]:
    """Verdict per candidate, plus a per-year class table per distinct candidate pool."""
    verdicts, tables = [], {}
    for name, cand in candidates.items():
        pool, sources = candidate_pool(sets, cand, id_column)
        early = [int(y) for y in cand["early_years"]]
        late = [int(y) for y in cand["late_years"]]
        invalid = 1.0
        if len(pool):
            years = valid_years(pool, first_year, last_year)
            invalid = float(years.isna().mean())
            key = ", ".join(sources)
            if key not in tables:
                table = pd.crosstab(years.dropna().astype(int), pool.loc[years.notna(), LABEL])
                tables[key] = table.rename(columns={0: "goodware", 1: "malware"})
        verdicts.append(
            RQ3Verdict(
                candidate=name,
                sources=sources,
                early_years=early,
                late_years=late,
                early_counts=_counts(window(pool, early, first_year, last_year)),
                late_counts=_counts(window(pool, late, first_year, last_year)),
                minimum=minimum,
                invalid_year_share=invalid,
            )
        )
    return verdicts, tables


def run_audit(
    sets: list[DadaSet],
    id_column: str,
    candidates: dict[str, dict[str, Any]],
    minimum: int,
    first_year: int = 2008,
    last_year: int = 2021,
) -> tuple[list[SetSummary], list[RQ3Verdict], dict[str, pd.DataFrame]]:
    summaries = [summarise(read_set(s, id_column), s.name, id_column) for s in sets]
    verdicts, tables = rq3_verdicts(sets, candidates, minimum, id_column, first_year, last_year)
    return summaries, verdicts, tables


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def _years(years: list[int]) -> str:
    return f"{years[0]}" if len(years) == 1 else f"{years[0]}–{years[-1]}"


def render_markdown(
    summaries: list[SetSummary],
    verdicts: list[RQ3Verdict],
    source: str,
    tables: dict[str, pd.DataFrame] | None = None,
) -> str:
    lines = [
        "# Data audit (roadmap step 1.4)",
        "",
        f"Generated by `shortcutcheck data audit` from `{source}`. Do not edit by hand.",
        "",
        "## All sets",
        "",
        "Year range uses the raw `year` column, including invalid values such as 1980.",
        "",
        "| Set | Rows | Unique hashes | Labelled | Unlabelled rows | Goodware | Malware "
        "| Malware ratio | Year min | Year max |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.name} | {s.rows} | {s.unique_hashes} | {'yes' if s.labelled else 'no'} "
            f"| {s.unlabelled_rows} | {s.goodware} | {s.malware} "
            f"| {_fmt(s.malware_ratio, 3)} | {_fmt(s.year_min)} | {_fmt(s.year_max)} |"
        )
    lines += [
        "",
        "## RQ3 sufficiency (Check for step 1.4)",
        "",
        "Classes are counted by each app's own `year` inside the early and late windows, "
        "after pooling the family's apps and removing duplicate hashes. Set names such as "
        "ACT14/ACT17 are time barriers, not app years, so they are not used for this check.",
        "",
        "| Candidate | Sources | Early | Early good/mal | Late | Late good/mal "
        "| Invalid year | Minimum | Result |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for v in verdicts:
        lines.append(
            f"| {v.candidate} | {', '.join(v.sources) or 'none matched'} "
            f"| {_years(v.early_years)} | {v.early_counts[0]}/{v.early_counts[1]} "
            f"| {_years(v.late_years)} | {v.late_counts[0]}/{v.late_counts[1]} "
            f"| {v.invalid_year_share:.1%} | {v.minimum} | {'PASS' if v.passes else 'FAIL'} |"
        )
    passing = [v.candidate for v in verdicts if v.passes]
    lines.append("")
    if passing:
        lines.append(f"**Check: PASS for {', '.join(passing)}.**")
    else:
        lines.append(
            "**Check: FAIL.** No candidate has enough labelled apps of both classes in both "
            "windows. Redesign RQ3 now, before Gate 1 (roadmap 1.4), e.g. the "
            "VirusShare-per-year fallback, and record why in the methodology chapter."
        )
    failing = [v.candidate for v in verdicts if not v.passes]
    if passing and failing:
        lines.append(f"Failing candidates: {', '.join(failing)}.")
    for key, table in (tables or {}).items():
        lines += ["", f"### Apps per year: {key}", "", "| Year | Goodware | Malware |"]
        lines.append("| ---: | ---: | ---: |")
        for year, row in table.iterrows():
            lines.append(
                f"| {year} | {int(row.get('goodware', 0))} | {int(row.get('malware', 0))} |"
            )
    lines.append("")
    return "\n".join(lines)
