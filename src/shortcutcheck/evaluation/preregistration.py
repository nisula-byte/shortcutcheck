"""Roadmap step 3.2 Check, and rule 3 ("if you change the analysis afterwards, report both").

* ``check_preregistration``: PREREGISTRATION.md must be committed before the first
  ``load_test()`` call for the RQ3 split (the 2017-19 labels: the roadmap's Check) and, as this
  project also pre-registers H1 and H2, before every other test access.
* ``analysis_status``: whether the analysis code (``src/shortcutcheck/analysis``) still equals
  the version committed with the pre-registration. Every hypothesis report prints this.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from shortcutcheck.config import Config
from shortcutcheck.evaluation.locked_test import access_log

PREREG_FILE = "PREREGISTRATION.md"
ANALYSIS_PATHS = ("src/shortcutcheck/analysis",)


def _git(root: Path, *args: str) -> str:
    """Output of a git command, or "" when ``root`` is not a git repository."""
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return done.stdout.strip()


def preregistration_commit(root: Path) -> tuple[str, datetime] | None:
    """The first commit that added PREREGISTRATION.md, and its commit time."""
    out = _git(root, "log", "--diff-filter=A", "--format=%H %cI", "--", PREREG_FILE)
    if not out:
        return None
    sha, when = out.splitlines()[-1].split()
    return sha, datetime.fromisoformat(when)


@dataclass(frozen=True)
class PreregCheck:
    commit: str | None
    committed_at: datetime | None
    rq3_split: str
    first_rq3_access: datetime | None
    early_accesses: list[str] = field(default_factory=list)  # any split, before the commit

    @property
    def passes(self) -> bool:
        return self.commit is not None and not self.early_accesses

    def describe(self) -> str:
        if self.commit is None:
            return f"FAIL: {PREREG_FILE} is not committed."
        lines = [
            f"{PREREG_FILE} first committed in `{self.commit[:7]}` at "
            f"{self.committed_at.isoformat() if self.committed_at else '?'}.",
            f"First load_test() access to `{self.rq3_split}` (2017-19): "
            + (self.first_rq3_access.isoformat() if self.first_rq3_access else "none yet")
            + ".",
        ]
        if self.early_accesses:
            lines.append(
                "Test accesses BEFORE the pre-registration: " + "; ".join(self.early_accesses)
            )
        lines.append(f"Check: {'PASS' if self.passes else 'FAIL'}")
        return "\n".join(lines)


def check_preregistration(cfg: Config) -> PreregCheck:
    rq3 = f"rq3-{cfg.rq3_primary}"
    found = preregistration_commit(cfg.root)
    log = access_log(cfg)
    rq3_times = sorted(datetime.fromisoformat(str(e["utc"])) for e in log if e["split"] == rq3)
    if found is None:
        return PreregCheck(None, None, rq3, rq3_times[0] if rq3_times else None)
    sha, when = found
    early = [
        f"{e['split']} at {e['utc']}" for e in log if datetime.fromisoformat(str(e["utc"])) <= when
    ]
    return PreregCheck(sha, when, rq3, rq3_times[0] if rq3_times else None, early)


def analysis_status(root: Path) -> str:
    """One line for every hypothesis report: is the analysis code the pre-registered version?"""
    found = preregistration_commit(root)
    if found is None:
        return "Analysis code: not yet pre-registered (no PREREGISTRATION.md commit)."
    sha = found[0]
    changed = _git(root, "diff", "--name-only", sha, "--", *ANALYSIS_PATHS)
    if not changed:
        return f"Analysis code: identical to the pre-registered version (`{sha[:7]}`)."
    files = ", ".join(f"`{f}`" for f in changed.splitlines())
    return (
        f"Analysis code: **CHANGED since the pre-registered version** (`{sha[:7]}`): {files}. "
        "Rule 3: report the result under both versions."
    )
