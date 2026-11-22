"""ShortcutCheck dashboard (roadmap 4.3): a Streamlit view of a gate report.

It only reads JSON reports written by ``shortcutcheck run --report`` or the pytest plugin's
``--shortcutcheck-reports``; nothing is hard-coded. Start it with
``shortcutcheck dashboard report.json`` (or ``streamlit run dashboard_app.py -- report.json``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from shortcutcheck.gate import GateError, GateReport

ENV = "SHORTCUTCHECK_REPORT"
BAR = "#2a78d6"
RULE = "#6b6a64"


def report_path(argv: list[str] | None = None) -> Path | None:
    """The report named after ``--`` on the command line, or in $SHORTCUTCHECK_REPORT."""
    if argv is None:
        # Only `streamlit run dashboard_app.py -- report.json` puts the report in sys.argv.
        launched = bool(sys.argv) and Path(sys.argv[0]).name == "dashboard_app.py"
        argv = sys.argv[1:] if launched else []
    args = argv
    if args:
        return Path(args[0])
    env = os.environ.get(ENV)
    return Path(env) if env else None


def read_report(source: Path | bytes) -> GateReport:
    if isinstance(source, Path):
        return GateReport.read(source)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "shortcutcheck-uploaded-report.json"
    tmp.write_bytes(source)
    return GateReport.read(tmp)


def mr_frame(report: GateReport) -> pd.DataFrame:
    rows = []
    for r in report.mrs:
        rows.append(
            {
                "relation": r.mr,
                "what changes": r.description,
                "rate": r.rate if r.applicable else None,
                "ci_low": r.ci_low if r.applicable else None,
                "ci_high": r.ci_high if r.applicable else None,
                "counted": r.n_counted,
                "flips": r.n_violations,
                "unrealistic": r.n_unrealistic,
                "result": (
                    "not applicable"
                    if not r.applicable
                    else ("pass" if r.passed else f"FAIL (> {report.threshold:g})")
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_frame(report: GateReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": f.feature, "rate": f.rate, "counted": f.n_counted, "flips": f.n_violations}
            for f in report.feature_rates
            if f.n_counted
        ],
        columns=["field", "rate", "counted", "flips"],
    )


def examples_frame(report: GateReport, mr: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "app": e.app_id,
                "row": e.row,
                "p(malware) before": round(e.p_before, 3),
                "p(malware) after": round(e.p_after, 3),
                "fields changed": ", ".join(e.changes),
            }
            for e in report.examples
            if e.mr == mr
        ],
        columns=["app", "row", "p(malware) before", "p(malware) after", "fields changed"],
    )


def change_frame(report: GateReport, mr: str, app: str) -> pd.DataFrame:
    for e in report.examples:
        if e.mr == mr and e.app_id == app:
            return pd.DataFrame(
                [{"field": k, "before": v[0], "after": v[1]} for k, v in e.changes.items()]
            )
    return pd.DataFrame(columns=["field", "before", "after"])


def rate_chart(frame: pd.DataFrame, threshold: float) -> Any:
    import altair as alt

    data = frame.dropna(subset=["rate"])
    top = max(0.1, float(data["rate"].max() if len(data) else 0.0) * 1.15)
    base = alt.Chart(data).encode(
        x=alt.X("relation:N", title=None, sort=None, axis=alt.Axis(labelAngle=0))
    )
    bars = base.mark_bar(color=BAR, cornerRadiusEnd=4, size=28).encode(
        y=alt.Y("rate:Q", title="violation rate", scale=alt.Scale(domain=[0, top])),
        tooltip=[
            "relation",
            "what changes",
            alt.Tooltip("rate:Q", format=".3f"),
            "counted",
            "flips",
            "result",
        ],
    )
    whiskers = base.mark_rule(color="#2b2b29").encode(y="ci_low:Q", y2="ci_high:Q")
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color=RULE, strokeDash=[4, 4])
        .encode(y="threshold:Q")
    )
    return (bars + whiskers + rule).properties(height=260)


def feature_chart(frame: pd.DataFrame) -> Any:
    import altair as alt

    return (
        alt.Chart(frame)
        .mark_bar(color=BAR, cornerRadiusEnd=4, size=20)
        .encode(
            y=alt.Y("field:N", title=None, sort="-x"),
            x=alt.X("rate:Q", title="MR4 violation rate when only this field changes"),
            tooltip=["field", alt.Tooltip("rate:Q", format=".3f"), "counted", "flips"],
        )
        .properties(height=max(80, 36 * len(frame)))
    )


def main() -> None:  # pragma: no cover - exercised through streamlit's AppTest
    import streamlit as st

    st.set_page_config(page_title="ShortcutCheck report", layout="wide")
    st.title("ShortcutCheck report")
    path = report_path()
    report: GateReport | None = None
    try:
        if path is not None:
            report = read_report(path)
        else:
            upload = st.file_uploader(
                "Open a report written by `shortcutcheck run --report`", type="json"
            )
            if upload is not None:
                report = read_report(upload.getvalue())
    except (OSError, GateError, json.JSONDecodeError, KeyError) as err:
        st.error(f"Could not read the report: {err}")
        return
    if report is None:
        st.info("No report loaded. Run `shortcutcheck dashboard report.json` or upload one.")
        return

    if report.passed:
        st.success(f"PASS: every relation is at or below the threshold {report.threshold:g}.")
    else:
        st.error(
            f"FAIL: {', '.join(report.failing)} exceed the threshold {report.threshold:g}. "
            "The model's decisions depend on metadata."
        )
    st.caption(
        f"Model `{report.model}` · inputs `{report.data}` ({report.n_checked} of "
        f"{report.n_inputs} checked) · training data `{report.train}` · realism check "
        f"{'on' if report.realism else 'off'} · seed {report.seed} · ShortcutCheck "
        f"{report.version} · {report.created_utc}"
    )

    frame = mr_frame(report)
    cols = st.columns(max(1, len(frame)))
    for col, row in zip(cols, frame.to_dict("records"), strict=False):
        value = "n/a" if row["rate"] is None or pd.isna(row["rate"]) else f"{row['rate']:.3f}"
        col.metric(f"{row['relation']} violation rate", value, help=str(row["what changes"]))

    st.subheader("Violation rate per relation")
    st.caption(
        "Bars: share of checked apps whose prediction changed; lines: Wilson 95% interval; "
        "dashed: threshold."
    )
    st.altair_chart(rate_chart(frame, report.threshold), use_container_width=True)
    st.dataframe(frame, hide_index=True, use_container_width=True)

    features = feature_frame(report)
    if len(features):
        st.subheader("Which metadata field the model leans on")
        st.caption(
            "MR4 restricted to one field: how often the prediction changes when only that "
            "field takes another class's value."
        )
        st.altair_chart(feature_chart(features), use_container_width=True)

    st.subheader("Apps whose prediction changed")
    with_examples = [r.mr for r in report.mrs if any(e.mr == r.mr for e in report.examples)]
    if not with_examples:
        st.write("No prediction changed under any relation.")
        return
    mr = st.selectbox("Relation", with_examples)
    examples = examples_frame(report, mr)
    st.dataframe(examples, hide_index=True, use_container_width=True)
    app = st.selectbox("Inspect an app", examples["app"].tolist())
    st.dataframe(change_frame(report, mr, app), hide_index=True, use_container_width=True)
