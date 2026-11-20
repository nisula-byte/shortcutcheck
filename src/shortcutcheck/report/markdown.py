"""Turn MR runs into a Markdown table (roadmap step 2.1; extended with CIs in 2.6)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from shortcutcheck.runner.engine import MRRun


def _fmt(value: float, digits: int = 3) -> str:
    return "–" if math.isnan(value) else f"{value:.{digits}f}"


def render_runs(runs: Iterable[MRRun], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Counted = inputs the MR changed and whose new values are valid. "
        "Violation rate = share of counted inputs whose predicted class flipped.",
        "",
        "| MR | Seed | Inputs | Perturbed | Invalid | Counted | Violations | Violation rate "
        "| Mean abs. change in p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            f"| {run.mr_id} | {run.seed} | {run.n_inputs} | {int(run.perturbed.sum())} "
            f"| {int((run.perturbed & ~run.valid).sum())} | {run.n_counted} "
            f"| {run.n_violations} | {_fmt(run.violation_rate)} | {_fmt(run.mean_abs_delta)} |"
        )
    return "\n".join(lines) + "\n"
