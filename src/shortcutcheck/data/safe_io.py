"""The only sanctioned way to write files in this project.

Every writer goes through :func:`guard_write`, which refuses any path inside ``data/raw``.
This enforces the step 1.3 Check ("no code writes to data/raw") at run time; a test also
scans the source tree for direct writes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class RawDataWriteError(PermissionError):
    """Raised when code tries to write inside data/raw."""


def guard_write(path: Path, raw_root: Path) -> Path:
    resolved = path.resolve()
    raw = raw_root.resolve()
    if resolved == raw or raw in resolved.parents:
        raise RawDataWriteError(f"Refusing to write inside data/raw: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_text(path: Path, text: str, raw_root: Path) -> Path:
    target = guard_write(path, raw_root)
    target.write_text(text, encoding="utf-8")
    return target


def write_frame(frame: pd.DataFrame, path: Path, raw_root: Path) -> Path:
    target = guard_write(path, raw_root)
    frame.to_csv(target, index=False)
    return target
