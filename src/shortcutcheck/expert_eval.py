"""Step 4.6: analysis of the expert evaluation questionnaire (docs/evaluation/).

Computes the median and interquartile range of each construct (usefulness U1-U3,
understandability C1-C3, CI fit I1-I3), the share of correct task interpretations, and the
NFR8 criterion (median usefulness at least 4 of 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CONSTRUCTS = {
    "usefulness": ("U1", "U2", "U3"),
    "understandability": ("C1", "C2", "C3"),
    "CI fit": ("I1", "I2", "I3"),
}
TASKS = ("task1", "task2")
NFR8_MEDIAN = 4.0
MIN_PARTICIPANTS = 5


@dataclass(frozen=True)
class Construct:
    name: str
    median: float
    q1: float
    q3: float
    n: int


@dataclass(frozen=True)
class EvalResult:
    n_participants: int
    constructs: list[Construct]
    task_correct: dict[str, float]

    @property
    def usefulness_median(self) -> float:
        return next(c.median for c in self.constructs if c.name == "usefulness")

    @property
    def nfr8(self) -> bool:
        return self.usefulness_median >= NFR8_MEDIAN

    @property
    def enough_participants(self) -> bool:
        return self.n_participants >= MIN_PARTICIPANTS


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    items = [i for group in CONSTRUCTS.values() for i in group]
    missing = [c for c in ("participant", *items, *TASKS) if c not in frame.columns]
    if missing:
        raise ValueError(f"responses are missing columns {missing}")
    if frame["participant"].duplicated().any():
        raise ValueError("a participant code appears twice")
    values = frame[items].stack()
    if not ((values >= 1) & (values <= 5)).all():
        raise ValueError("ratings must be integers from 1 to 5")
    bad = set(frame[list(TASKS)].stack()) - {"correct", "partly", "incorrect"}
    if bad:
        raise ValueError(f"task outcomes must be correct, partly or incorrect, not {sorted(bad)}")
    return frame


def analyse(frame: pd.DataFrame) -> EvalResult:
    validate(frame)
    constructs = []
    for name, items in CONSTRUCTS.items():
        # One score per participant: the median of their construct items.
        scores = frame[list(items)].median(axis=1)
        constructs.append(
            Construct(
                name,
                float(scores.median()),
                float(scores.quantile(0.25)),
                float(scores.quantile(0.75)),
                int(scores.notna().sum()),
            )
        )
    tasks = {t: float((frame[t] == "correct").mean()) for t in TASKS}
    return EvalResult(len(frame), constructs, tasks)


def render(result: EvalResult) -> str:
    lines = [
        "# Expert evaluation results (roadmap 4.6)",
        "",
        f"Participants: {result.n_participants}"
        + ("" if result.enough_participants else f" (fewer than the planned {MIN_PARTICIPANTS})"),
        "",
        "| Construct | Median | IQR | n |",
        "| --- | ---: | --- | ---: |",
    ]
    for c in result.constructs:
        lines.append(f"| {c.name} | {c.median:.1f} | {c.q1:.1f}–{c.q3:.1f} | {c.n} |")
    lines += ["", "| Task | Interpreted correctly |", "| --- | ---: |"]
    for t, share in result.task_correct.items():
        lines.append(f"| {t} | {share:.0%} |")
    lines += [
        "",
        f"**NFR8 (median usefulness >= {NFR8_MEDIAN:g}): "
        f"{'met' if result.nfr8 else 'not met'}** (median {result.usefulness_median:.1f}).",
        "",
    ]
    return "\n".join(lines)
