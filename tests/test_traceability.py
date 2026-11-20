"""Roadmap step 2.8 Check: every requirement has a test or evaluation step that verifies it.

Reads docs/requirements.yaml and fails if a requirement has no verifier, if a "done"
requirement has no test, or if a named test, report or roadmap step does not exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from shortcutcheck.mrs.applicability import render_markdown as render_applicability
from shortcutcheck.mrs.realism_report import render_markdown as render_realism
from shortcutcheck.runner.known_answers import render_markdown as render_known_answers

REPO = Path(__file__).resolve().parents[1]
ROADMAP_STEPS = {f"3.{i}" for i in range(1, 9)} | {f"4.{i}" for i in range(1, 8)}


def requirements() -> list[dict[str, object]]:
    data = yaml.safe_load((REPO / "docs" / "requirements.yaml").read_text(encoding="utf-8"))
    return [*data["functional"], *data["non_functional"]]


def defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


def test_every_requirement_has_an_existing_verifier() -> None:
    reqs = requirements()
    ids = [str(r["id"]) for r in reqs]
    assert len(ids) == len(set(ids)), "requirement ids must be unique"
    problems = []
    for r in reqs:
        verifiers = [str(v) for v in r.get("verified_by") or []]  # type: ignore[attr-defined]
        if not verifiers:
            problems.append(f"{r['id']}: no verifier")
        if r["status"] == "done" and not any(v.startswith(("test:", "ci:")) for v in verifiers):
            problems.append(f"{r['id']}: done but verified by no test or CI step")
        if r["status"] not in ("done", "planned") or r["priority"] not in (
            "must",
            "should",
            "could",
        ):
            problems.append(f"{r['id']}: bad status or priority")
        for v in verifiers:
            kind, _, target = v.partition(":")
            if kind == "test":
                file, _, func = target.partition("::")
                path = REPO / file
                if not path.exists() or func not in defined_tests(path):
                    problems.append(f"{r['id']}: no test {target}")
            elif kind == "report":
                if not (REPO / target).exists():
                    problems.append(f"{r['id']}: no report {target}")
            elif kind == "roadmap":
                if target not in ROADMAP_STEPS:
                    problems.append(f"{r['id']}: unknown roadmap step {target}")
            elif kind != "ci":
                problems.append(f"{r['id']}: unknown verifier kind {v}")
    assert not problems, "\n".join(problems)


def test_reports_name_their_git_commit() -> None:
    """NFR4: every generated report states the commit that produced it."""
    commit = "0123456789abcdef"
    for text in (
        render_applicability([], commit),
        render_realism([], commit),
        render_known_answers([], commit),
    ):
        assert f"`{commit}`" in text
    pilot = (REPO / "reports" / "pilot_rq1_rq12-DR-AG.md").read_text(encoding="utf-8")
    assert "Git commit `" in pilot
    rows = (REPO / "reports" / "pilot_rq1_rq12-DR-AG.csv").read_text(encoding="utf-8")
    assert "mlflow_run" in rows.splitlines()[0]
