"""Pre-registered addenda (PREREGISTRATION-ADDENDUM-*.md). Kept apart from ``analysis`` so the
H1-H3 analysis code stays identical to its pre-registered version."""

from __future__ import annotations

import subprocess
from pathlib import Path


class AddendumNotCommittedError(RuntimeError):
    """An addendum analysis was run before its pre-registration was committed unchanged."""


def require_committed(root: Path, filename: str) -> str:
    """Return the commit that added ``filename``; raise unless it is committed and unchanged."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    added = git("log", "--diff-filter=A", "--format=%H", "--", filename)
    if added.returncode != 0 or not added.stdout.strip():
        raise AddendumNotCommittedError(f"{filename} is not committed: commit it before running.")
    if git("diff", "--quiet", "HEAD", "--", filename).returncode != 0:
        raise AddendumNotCommittedError(f"{filename} has uncommitted changes.")
    return added.stdout.strip().splitlines()[-1]
