"""The locked test set (strict-check rule 2).

``load_test`` is the only function that reads test data. Every call appends one JSON line to
``logs/test_access.jsonl`` with a UTC timestamp, the split, the stated purpose and the git
commit. Commit that log: it is the evidence for "test labels looked at once per experiment"
and for the step 3.2 check that pre-registration predates the first 2017 access.

Only modules inside ``shortcutcheck.evaluation`` may import this file; a test enforces it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from shortcutcheck.config import Config
from shortcutcheck.data.splits import read_index, split_paths
from shortcutcheck.provenance import git_commit

LOG_NAME = "test_access.jsonl"


def load_test(cfg: Config, split_name: str, *, purpose: str) -> pd.DataFrame:
    if not purpose.strip():
        raise ValueError("State why the test set is being opened (purpose=...).")
    frame = pd.read_csv(
        split_paths(cfg, split_name).test, low_memory=False, dtype={cfg.id_column: str}
    )
    if set(frame[cfg.id_column]) != read_index(cfg, split_name, "test"):
        raise ValueError(f"Test data for '{split_name}' does not match its frozen index.")
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split_name,
        "purpose": purpose,
        "rows": len(frame),
        "git_commit": git_commit(cfg.root),
    }
    with (cfg.logs_dir / LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return frame


def access_log(cfg: Config) -> list[dict[str, object]]:
    path = cfg.logs_dir / LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
