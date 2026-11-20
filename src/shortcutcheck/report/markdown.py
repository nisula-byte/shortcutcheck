"""Turn MR runs into a Markdown table (roadmap step 2.1; extended with CIs in 2.6)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from shortcutcheck.runner.engine import MRRun
from shortcutcheck.runner.stats import fmt_rate, summarise


def _fmt(value: float, digits: int = 3) -> str:
    return "–" if math.isnan(value) else f"{value:.{digits}f}"


def render_runs(runs: Iterable[MRRun], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Counted = inputs the MR changed whose new values are valid and realistic. "
        "Violation rate = share of counted inputs that violate the MR, with a 95% bootstrap CI "
        "(Wilson when there are no violations, or only violations).",
        "",
        "| MR | Seed | Inputs | Perturbed | Invalid | Unrealistic | Counted | Violations "
        "| Violation rate [95% CI] | Without realism filter | Mean abs. change in p [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        s = summarise(run, seed=run.seed)
        delta = (
            "–"
            if math.isnan(s.delta)
            else f"{s.delta:.3f} [{s.delta_ci.low:.3f}, {s.delta_ci.high:.3f}]"
        )
        lines.append(
            f"| {run.mr_id} | {run.seed} | {run.n_inputs} | {int(run.perturbed.sum())} "
            f"| {int((run.perturbed & ~run.valid).sum())} | {run.n_unrealistic} "
            f"| {run.n_counted} | {run.n_violations} | {fmt_rate(s)} "
            f"| {_fmt(run.violation_rate_unfiltered)} | {delta} |"
        )
    return "\n".join(lines) + "\n"
